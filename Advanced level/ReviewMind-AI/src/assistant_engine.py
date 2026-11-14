"""Local, rule-based question answering for ReviewMind AI.

The assistant answers only from the uploaded DataFrame and from analysis
results that have already been computed. It never runs a model, calls an
external service, or executes arbitrary code.

Every answer returned by :func:`answer_question` has the same structure::

    {
        "question":  str,
        "intent":    str,
        "available": bool,          # False when the data/analysis is missing
        "answer":    str,
        "evidence":  dict,          # raw verified numbers
        "tables":    [{"title": str, "rows": [dict, ...]}],
        "charts":    [{"type": str, "title": str, "data": [...], ...}],
    }
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from .data_profiler import detect_rating_columns, parse_dates


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "about", "after", "again", "also", "and", "any", "are", "because", "been",
    "being", "but", "can", "could", "customer", "customers", "did", "didn",
    "does", "doesn", "don", "for", "from", "get", "got", "had", "has", "have",
    "how", "into", "its", "just", "more", "most", "not", "one", "our", "out",
    "reviews", "review", "should", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "too", "very", "was",
    "wasn", "were", "what", "when", "where", "which", "while", "will", "with",
    "would", "why", "you", "your",
}

CATEGORY_ORDER = (
    "Dataset Overview",
    "Sentiment Analysis",
    "Emotion Analysis",
    "Topics and Keywords",
    "Customer Complaints",
    "Business Insights",
)

# Keys that belong to one dataset view. They are discarded when the dataset,
# worksheet, or header row changes. The original DataFrame is never listed here.
_STALE_STATE_KEYS = (
    "analysis_results",
    "analysis_text_column",
    "analysis_config_signature",
    "assistant_dataset",
    "assistant_metadata",
    "assistant_history",
    "assistant_context_cache",
    "assistant_question",
    "chat_category",
    "chat_minimized",
    "chat_closed",
    "custom_graph_result",
    "custom_graph_date_column",
    "custom_graph_value_column",
    "custom_graph_chart_type",
    "custom_graph_aggregation",
    "custom_graph_date_grouping",
    "custom_graph_group_for",
    "selected_visualization",
    "selected_text",
    "selected_label",
    "analysis_mode",
    "analysis_model",
    "analysis_sentiment",
    "analysis_emotion",
    "analysis_topics",
    "analysis_evaluation",
    "guided_quick_insights",
    "guided_explore_data",
    "guided_last_question",
)


# ---------------------------------------------------------------------------
# Question normalisation and intent detection
# ---------------------------------------------------------------------------

def normalize_question(question: str) -> str:
    """Normalize a user question for deterministic rule matching."""

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")
    return " ".join(
        re.sub(r"[^a-z0-9%./_-]+", " ", question.lower()).split()
    )


def _words(normalized: str) -> set[str]:
    return {word.strip(".-_/") for word in normalized.split()}


def _fallback_intent(normalized: str) -> str:
    """Whole-word / phrase rules for questions that are not guided questions."""

    words = _words(normalized)

    def any_word(*tokens: str) -> bool:
        return bool(words.intersection(tokens))

    if any_word("trend", "trends", "monthly", "timeline") or "over time" in normalized:
        return "date_trends"
    if any_word("eda") or "exploratory" in normalized:
        return "eda"
    if any_word("compare", "comparison") and any_word("sentiment"):
        return "category_compare"
    if any_word("repeatedly", "recurring", "repeated", "repeat", "repeats"):
        return "recurring_issues"
    if any_word("focus", "prioritize", "prioritise", "priority"):
        return "focus_first"
    if any_word("improve", "improvement", "improvements", "improving"):
        return "improvement"
    if any_word("complaint", "complaints", "unhappy", "problems", "issues"):
        return "complaints"
    if any_word("quality", "duplicate", "duplicates", "blank", "missing"):
        return "data_quality"
    if any_word("insight", "insights") and any_word("rating", "ratings", "star", "stars", "score", "scores"):
        return "rating_insights"
    if any_word("rating", "ratings", "star", "stars"):
        return "rating_stats"
    if any_word("keyword", "keywords") or "most common word" in normalized:
        return "keywords"
    if any_word("topic", "topics", "theme", "themes"):
        return "topics"
    if any_word("dominant") and any_word("emotion", "emotions"):
        return "emotion_dominant"
    if any_word("emotion", "emotions", "joy", "sadness", "anger", "fear", "surprise"):
        if any_word("show", "chart", "plot", "graph", "visualize", "visualise"):
            return "emotion_chart"
        return "emotion"
    if any_word("sentiment", "positive", "negative", "positives", "negatives"):
        if "positive feedback" in normalized:
            return "positive_areas"
        if "most negative" in normalized or "negative reviews" in normalized and "how many" not in normalized:
            return "negative_reviews"
        if "most positive" in normalized or "positive reviews" in normalized and "how many" not in normalized:
            return "positive_reviews"
        if "how many" in normalized or "count" in normalized or "number of" in normalized:
            return "sentiment_counts"
        if any_word("show", "chart", "plot", "graph", "visualize", "visualise"):
            return "sentiment_chart"
        return "sentiment"
    if any_word("summarize", "summarise", "summary", "findings", "business", "insights"):
        return "business_insights"
    if any_word("correlation", "numeric", "numerical", "average", "mean",
                "median", "minimum", "maximum"):
        return "numeric_stats"
    if any_word("columns", "column") and not any_word("rows", "row"):
        return "columns"
    if any_word("records") and "how many" in normalized:
        return "record_count"
    if (
        any_word("row", "rows", "column", "columns", "dataset", "records", "data")
    ):
        return "overview"
    return "unsupported"


# Guided questions are matched exactly (after normalisation), so the chat never
# depends on fuzzy keyword matching. Each entry is
# (category, question, intent, requirement(s), reason).
_QUESTION_SPECS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    ("Dataset Overview", "How many records are in this dataset?",
     "record_count", ("data",), "Available for every non-empty dataset."),
    ("Dataset Overview", "Which columns are available?",
     "columns", ("data",), "Available for every non-empty dataset."),
    ("Dataset Overview", "What are the important data-quality issues?",
     "data_quality", ("data",), "Checks the uploaded data itself."),
    ("Dataset Overview", "Perform basic EDA.",
     "eda", ("eda",), "Suitable numeric, categorical, or date columns exist."),
    ("Dataset Overview", "Calculate the average, minimum, maximum, and median.",
     "numeric_stats", ("numeric",), "Numeric columns are available."),
    ("Dataset Overview", "Show the monthly trend of records.",
     "date_trends", ("date",), "A valid date column is available."),

    ("Sentiment Analysis", "What is the overall sentiment distribution?",
     "sentiment", ("sentiment",), "Uses completed sentiment predictions."),
    ("Sentiment Analysis", "How many positive and negative reviews are there?",
     "sentiment_counts", ("sentiment",), "Uses completed sentiment predictions."),
    ("Sentiment Analysis", "Show the sentiment distribution.",
     "sentiment_chart", ("sentiment",), "Uses completed sentiment predictions."),
    ("Sentiment Analysis", "Show monthly sentiment trends.",
     "date_trends", ("date_sentiment",),
     "Needs sentiment predictions and a date column for the same rows."),

    ("Emotion Analysis", "Which emotions appear most frequently?",
     "emotion", ("emotion",), "Uses completed emotion predictions."),
    ("Emotion Analysis", "What is the dominant emotion?",
     "emotion_dominant", ("emotion",), "Uses completed emotion predictions."),
    ("Emotion Analysis", "Show the emotion distribution.",
     "emotion_chart", ("emotion",), "Uses completed emotion predictions."),

    ("Topics and Keywords", "What are the most common topics?",
     "topics", ("topics",), "Uses completed topic discovery."),
    ("Topics and Keywords", "Which keywords appear most frequently?",
     "keywords", ("text",), "Counts terms in the selected feedback column."),

    ("Customer Complaints", "Show the most negative reviews.",
     "negative_reviews", ("negative_text",),
     "Uses analyzed records predicted as negative."),
    ("Customer Complaints", "What are the main customer complaints?",
     "complaints", ("negative_text",),
     "Uses negative predictions with topics or keywords."),
    ("Customer Complaints", "Which issues appear repeatedly?",
     "recurring_issues", ("negative_text",),
     "Counts terms shared by several negative reviews."),

    ("Business Insights", "What are the biggest areas for improvement?",
     "improvement", ("negative_text",),
     "Ranks areas by negative reviews."),
    ("Business Insights", "What should the business focus on first?",
     "focus_first", ("negative_text",),
     "Uses the area with the most negative reviews."),
    ("Business Insights", "What insights can be derived from the available ratings?",
     "rating_insights", ("rating",), "A valid rating column is available."),
    ("Business Insights", "What are the strongest positive feedback areas?",
     "positive_areas", ("positive_text",),
     "Uses analyzed records predicted as positive."),
    ("Business Insights", "Summarize the main customer feedback findings.",
     "business_insights", ("summary",),
     "Combines the verified findings currently available."),
)

_QUESTION_INTENTS = {
    normalize_question(question): intent
    for _, question, intent, _, _ in _QUESTION_SPECS
}


def detect_intent(question: str) -> str:
    """Return the supported intent matching ``question``."""

    normalized = normalize_question(question)
    if not normalized:
        return "unsupported"
    if normalized in _QUESTION_INTENTS:
        return _QUESTION_INTENTS[normalized]
    return _fallback_intent(normalized)


# ---------------------------------------------------------------------------
# Dataset identity and state reset
# ---------------------------------------------------------------------------

def build_dataset_signature(
    file_bytes: bytes,
    file_name: str,
    sheet_name: Optional[str] = None,
    header: int = 0,
) -> str:
    """Build a stable identity for a selected uploaded dataset view."""

    digest = hashlib.sha256(file_bytes).hexdigest()
    return "|".join((str(file_name), str(sheet_name), str(header), digest))


def reset_assistant_context(
    state: MutableMapping[str, Any],
    signature: str,
) -> bool:
    """Clear stale analysis and assistant values when the dataset changes.

    Returns ``True`` when the dataset view changed (state was reset) and
    ``False`` when ``signature`` matches the one already stored. The original
    uploaded DataFrame (``original_data``) is intentionally not touched here;
    the caller replaces it when this function returns ``True``.
    """

    if state.get("dataset_signature") == signature:
        return False
    for key in _STALE_STATE_KEYS:
        if key in state:
            del state[key]
    state["dataset_signature"] = signature
    return True


# ---------------------------------------------------------------------------
# Dataset context
# ---------------------------------------------------------------------------

def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse dates with the same rules the rest of the app uses.

    Delegates to :func:`src.data_profiler.parse_dates`, which infers
    day-first versus month-first numeric dates, treats blanks as missing,
    and returns timezone-naive values (unparseable values become NaT).
    """

    return parse_dates(series)


def _parseable_date_ratio(series: pd.Series, sample_size: int = 500) -> float:
    """Share of sampled non-blank values that look like dates."""

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return 0.0
    text = series.astype("string").str.strip()
    text = text[text.notna() & ~text.str.lower().isin(
        {"", "nan", "nat", "none", "null", "n/a", "na", "#n/a", "-", "--"}
    )]
    # Pure digits ("5", "2024") are ratings or ids, not dates.
    text = text[~text.str.fullmatch(r"\d+")]
    if text.empty:
        return 0.0
    if len(text) > sample_size:
        text = text.sample(sample_size, random_state=0)
    return float(parse_dates(text).notna().mean())


def _is_text_dtype(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _average_length(series: pd.Series, sample_size: int = 500) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    if len(non_null) > sample_size:
        non_null = non_null.sample(sample_size, random_state=0)
    return float(non_null.astype(str).str.len().mean())


def _detect_rating_columns(data: pd.DataFrame) -> list[str]:
    """Rating columns, using the project's detector with a safe fallback."""

    names: list[str] = []
    try:
        for item in detect_rating_columns(data) or []:
            name = item.get("column") if isinstance(item, Mapping) else item
            if name in data.columns and name not in names:
                names.append(name)
    except (KeyError, ValueError, TypeError):
        names = []
    return names


def get_dataset_context(
    data: pd.DataFrame,
    analysis_results: Optional[Mapping[str, Any]] = None,
    text_column: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the assistant context without replacing or mutating ``data``."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Dataset context requires a pandas DataFrame.")
    if data.empty:
        raise ValueError("The dataset is empty.")
    if text_column is not None and text_column not in data.columns:
        text_column = None

    numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    date_columns = [
        column for column in data.columns
        if column != text_column
        and column not in numeric
        and (
            pd.api.types.is_datetime64_any_dtype(data[column])
            or (_is_text_dtype(data[column]) and _parseable_date_ratio(data[column]) >= 0.8)
        )
    ]
    categorical = []
    for column in data.columns:
        if column in numeric or column in date_columns or column == text_column:
            continue
        series = data[column]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical.append(column)
        elif (
            _is_text_dtype(series)
            and series.nunique(dropna=True) <= 20
            and _average_length(series) <= 40
        ):
            categorical.append(column)
    text_columns = [
        column for column in data.columns
        if column not in date_columns
        and column not in numeric
        and _is_text_dtype(data[column])
    ]

    return {
        "data": data,
        "analysis_results": analysis_results or {},
        "text_column": text_column,
        "metadata": dict(metadata or {}),
        "columns": data.columns.tolist(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "rating_columns": _detect_rating_columns(data),
    }


def get_demo_dataset() -> pd.DataFrame:
    """Return a small, deterministic dataset for first-time exploration."""

    return pd.DataFrame({
        "review": [
            "Delivery was fast and the product quality was excellent.",
            "The item arrived damaged and support was slow to respond.",
            "Friendly support solved my issue quickly.",
            "The product is useful, but the packaging could improve.",
            "Very happy with the value and simple ordering process.",
            "The delivery was late and the instructions were unclear.",
            "Customer service was helpful and professional.",
            "The quality did not match the price I paid.",
        ],
        "rating": [5, 1, 4, 3, 5, 2, 4, 2],
        "category": [
            "delivery", "quality", "support", "packaging",
            "value", "delivery", "support", "quality"
        ],
        "date": [
            "2025-01-05", "2025-01-12", "2025-02-03", "2025-02-18",
            "2025-03-02", "2025-03-15", "2025-04-04", "2025-04-21"
        ],
    })


# ---------------------------------------------------------------------------
# Accessors for completed analysis results
# ---------------------------------------------------------------------------

def _context_data(context: Mapping[str, Any]) -> pd.DataFrame:
    data = context.get("data")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The assistant context has no dataset.")
    return data


def _results_dataframe(
    context: Mapping[str, Any],
) -> Optional[pd.DataFrame]:
    results = context.get("analysis_results") or {}
    value = results.get("results_dataframe")
    return value if isinstance(value, pd.DataFrame) else None


def _analysis(context: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    results = context.get("analysis_results") or {}
    analyses = results.get("analyses") or {}
    value = analyses.get(name)
    return value if isinstance(value, Mapping) else {}


def _rating_columns(context: Mapping[str, Any]) -> list[str]:
    columns = context.get("rating_columns")
    if columns is None:
        columns = _detect_rating_columns(_context_data(context))
    return list(columns)


def _aligned_column(
    context: Mapping[str, Any],
    column: Optional[str],
) -> Optional[pd.Series]:
    """Return ``column`` aligned with the analyzed rows, or ``None``.

    The values come from the analysis results when the column is stored there.
    Otherwise they come from the original data, but only after confirming that
    the analyzed rows are exactly the original rows with the same feedback text.
    """

    results = _results_dataframe(context)
    data = context.get("data")
    if results is None or not isinstance(data, pd.DataFrame) or column is None:
        return None
    if column in results.columns:
        return results[column]
    text = context.get("text_column")
    if column not in data.columns or text is None:
        return None
    if text not in results.columns or text not in data.columns:
        return None
    index = results.index
    if not (index.is_unique and data.index.is_unique and index.isin(data.index).all()):
        return None
    left = results[text].astype(str).to_numpy()
    right = data.loc[index, text].astype(str).to_numpy()
    if not np.array_equal(left, right):
        return None
    return data.loc[index, column]


def _label_key(label: Any) -> str:
    return str(label).strip().lower()


def _is_label(label: Any, polarity: str) -> bool:
    return _label_key(label).startswith(polarity[:3])


def _confidence_column(
    frame: pd.DataFrame,
    kind: str = "sentiment",
) -> Optional[str]:
    """Find a numeric confidence column for sentiment predictions, if any."""

    candidates = [
        column for column in frame.columns
        if "confidence" in str(column).lower()
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    preferred = [c for c in candidates if kind in str(c).lower()]
    others = [c for c in candidates if "emotion" not in str(c).lower()]
    for group in (preferred, others):
        if group:
            return group[0]
    return None


def _topic_keywords(context: Mapping[str, Any]) -> dict[Any, list[str]]:
    topics = _analysis(context, "topics").get("topics") or []
    return {
        item.get("topic"): list(item.get("keywords", []))
        for item in topics
        if isinstance(item, Mapping)
    }


def _capabilities(context: Mapping[str, Any]) -> dict[str, bool]:
    """Which kinds of questions the current data and analyses can support."""

    data = context.get("data")
    results = _results_dataframe(context)
    text_column = context.get("text_column")

    def has(column: str) -> bool:
        return (
            results is not None
            and column in results.columns
            and bool(results[column].notna().any())
        )

    sentiment = has("predicted_sentiment")
    emotion = has("predicted_emotion")
    topics = has("discovered_topic") and bool(_topic_keywords(context))
    text = bool(
        isinstance(data, pd.DataFrame)
        and text_column in getattr(data, "columns", [])
        and data[text_column].dropna().astype(str).str.strip().ne("").any()
    )
    aligned_text = _aligned_column(context, text_column) is not None
    date_columns = list(context.get("date_columns") or [])
    return {
        "data": isinstance(data, pd.DataFrame) and not data.empty,
        "sentiment": sentiment,
        "emotion": emotion,
        "topics": topics,
        "text": text,
        "negative_text": sentiment and aligned_text,
        "positive_text": sentiment and aligned_text,
        "rating": bool(_rating_columns(context)),
        "numeric": bool(context.get("numeric_columns")),
        "date": bool(date_columns),
        "date_sentiment": sentiment and any(
            _aligned_column(context, column) is not None
            for column in date_columns
        ),
        "eda": bool(
            context.get("numeric_columns")
            or context.get("categorical_columns")
            or date_columns
        ),
        "summary": sentiment or emotion or topics or text
        or bool(_rating_columns(context)),
    }


def get_guided_questions(
    context: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Return only questions the current data and analyses can answer.

    Categories keep the order of :data:`CATEGORY_ORDER`; empty categories are
    omitted. Each item has ``question``, ``reason``, and ``intent`` keys.
    """

    capabilities = _capabilities(context)
    grouped: dict[str, list[dict[str, str]]] = {
        category: [] for category in CATEGORY_ORDER
    }
    for category, question, intent, requirements, reason in _QUESTION_SPECS:
        if all(capabilities.get(name, False) for name in requirements):
            grouped[category].append({
                "question": question,
                "reason": reason,
                "intent": intent,
            })
    return {category: items for category, items in grouped.items() if items}


# ---------------------------------------------------------------------------
# Statistics helpers (all computed from supplied data only)
# ---------------------------------------------------------------------------

def _format_percentages(stats: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{label}: {percentage:.2f}%"
        for label, percentage in stats.get("percentages", {}).items()
    )


def _percentage_counts(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().astype(str).value_counts()
    total = int(counts.sum())
    return {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "percentages": {
            str(key): round(float(value / total * 100), 2)
            for key, value in counts.items()
        } if total else {},
        "total": total,
    }


def calculate_sentiment_statistics(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate statistics from completed predicted sentiment results."""

    data = _results_dataframe(context)
    if data is None or "predicted_sentiment" not in data.columns:
        return {"available": False, "reason": "Run sentiment analysis first."}
    statistics = _percentage_counts(data["predicted_sentiment"])
    if not statistics["total"]:
        return {"available": False, "reason": "No sentiment predictions are available."}
    return {"available": True, **statistics}


def calculate_emotion_statistics(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate statistics from completed predicted emotion results."""

    data = _results_dataframe(context)
    if data is None or "predicted_emotion" not in data.columns:
        return {"available": False, "reason": "Run emotion analysis first."}
    statistics = _percentage_counts(data["predicted_emotion"])
    if not statistics["total"]:
        return {"available": False, "reason": "No emotion predictions are available."}
    return {"available": True, **statistics}


def get_top_topics(context: Mapping[str, Any], limit: int = 5) -> dict[str, Any]:
    """Return topic keywords and observed document frequencies."""

    data = _results_dataframe(context)
    keywords = _topic_keywords(context)
    if data is None or "discovered_topic" not in data.columns or not keywords:
        return {"available": False, "reason": "Run topic discovery first."}

    counts = data["discovered_topic"].dropna().value_counts()
    if counts.empty:
        return {"available": False, "reason": "No topic assignments are available."}
    total = int(counts.sum())
    values = []
    for topic_id, count in counts.head(limit).items():
        key = topic_id.item() if hasattr(topic_id, "item") else topic_id
        values.append({
            "topic": key,
            "count": int(count),
            "percentage": round(float(count / total * 100), 2),
            "keywords": list(keywords.get(key, [])),
        })
    return {"available": True, "topics": values, "total": total}


def _keyword_counts(
    texts: pd.Series,
    limit: int,
    per_review: bool = False,
) -> list[tuple[str, int]]:
    """Count meaningful words. ``per_review`` counts reviews, not occurrences."""

    counter: Counter = Counter()
    for text in texts.dropna().astype(str):
        words = [
            word.strip("'-") for word in
            re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
        ]
        words = [word for word in words if len(word) > 2 and word not in _STOP_WORDS]
        counter.update(set(words) if per_review else words)
    return counter.most_common(limit)


def get_keyword_statistics(
    context: Mapping[str, Any],
    limit: int = 10,
) -> dict[str, Any]:
    """Count meaningful words in the selected feedback column."""

    column = context.get("text_column")
    data = context.get("data")
    if not isinstance(data, pd.DataFrame) or column not in data.columns:
        return {"available": False, "reason": "A valid text column is required."}
    counts = _keyword_counts(data[column], limit)
    if not counts:
        return {"available": False, "reason": "The feedback column has no usable text."}
    return {
        "available": True,
        "keywords": [
            {"keyword": word, "count": int(count)} for word, count in counts
        ],
    }


def _label_frame(
    context: Mapping[str, Any],
    polarity: str,
) -> tuple[Optional[pd.DataFrame], str]:
    """Analyzed rows predicted as ``polarity`` with their feedback text.

    Returns ``(frame, reason)``. ``frame`` is ``None`` when the analysis or the
    feedback text is unavailable; ``reason`` then explains why.
    """

    results = _results_dataframe(context)
    if results is None or "predicted_sentiment" not in results.columns:
        return None, "Run sentiment analysis first."
    text_column = context.get("text_column")
    text = _aligned_column(context, text_column)
    if text is None:
        return None, (
            "The selected feedback column could not be matched to the analyzed rows."
        )
    mask = results["predicted_sentiment"].map(lambda v: _is_label(v, polarity))
    frame = pd.DataFrame({"text": text[mask].astype(str)})
    confidence = _confidence_column(results)
    if confidence:
        frame["confidence"] = results.loc[mask, confidence]
    frame["row_index"] = frame.index
    if "confidence" in frame.columns:
        frame = frame.sort_values("confidence", ascending=False)
    return frame, ""


def _review_rows(
    context: Mapping[str, Any],
    polarity: str,
    limit: int,
) -> dict[str, Any]:
    frame, reason = _label_frame(context, polarity)
    if frame is None:
        return {"available": False, "reason": reason}
    shown = frame.head(limit)
    reviews = []
    for _, row in shown.iterrows():
        item = {"row_index": row["row_index"], "text": str(row["text"])}
        if "confidence" in shown.columns and pd.notna(row["confidence"]):
            item["confidence"] = round(float(row["confidence"]), 3)
        reviews.append(item)
    for item in reviews:
        if hasattr(item["row_index"], "item"):
            item["row_index"] = item["row_index"].item()
    return {
        "available": True,
        "polarity": polarity,
        "total": int(len(frame)),
        "ranked_by_confidence": "confidence" in frame.columns,
        "reviews": reviews,
    }


def get_negative_reviews(
    context: Mapping[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    """Return actual analyzed reviews predicted as negative."""

    return _review_rows(context, "negative", limit)


def get_positive_reviews(
    context: Mapping[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    """Return actual analyzed reviews predicted as positive."""

    return _review_rows(context, "positive", limit)


def _label_areas(
    context: Mapping[str, Any],
    polarity: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Rank discussion areas among reviews predicted as ``polarity``.

    Areas are discovered topics when topic discovery has been run; otherwise
    they are keywords counted once per review.
    """

    frame, reason = _label_frame(context, polarity)
    if frame is None:
        return {"available": False, "reason": reason}
    total = int(len(frame))
    if total == 0:
        return {"available": True, "label_total": 0, "basis": None, "areas": []}

    results = _results_dataframe(context)
    keywords = _topic_keywords(context)
    if results is not None and "discovered_topic" in results.columns and keywords:
        topic_series = results["discovered_topic"]
        in_label = topic_series.loc[frame.index].dropna()
        label_counts = in_label.value_counts()
        all_counts = topic_series.dropna().value_counts()
        areas = []
        for topic_id, count in label_counts.head(limit).items():
            key = topic_id.item() if hasattr(topic_id, "item") else topic_id
            records = int(all_counts.get(topic_id, 0))
            areas.append({
                "label": f"Topic {key}",
                "keywords": list(keywords.get(key, [])),
                "reviews": int(count),
                "records": records,
                "rate": round(float(count / records * 100), 2) if records else 0.0,
            })
        return {"available": True, "label_total": total, "basis": "topic",
                "areas": areas}

    counts = _keyword_counts(frame["text"], limit, per_review=True)
    areas = [
        {"label": word, "keywords": [word], "reviews": int(count),
         "share": round(float(count / total * 100), 2)}
        for word, count in counts
    ]
    return {"available": True, "label_total": total, "basis": "keyword",
            "areas": areas}


# ---------------------------------------------------------------------------
# Chart and table specifications
# ---------------------------------------------------------------------------

def _chart(
    chart_type: str,
    title: str,
    rows: list[dict[str, Any]],
    x: str,
    y: str,
) -> dict[str, Any]:
    return {"type": chart_type, "title": title, "data": rows, "x": x, "y": y}


def _distribution_chart(
    stats: Mapping[str, Any],
    title: str,
    chart_type: str = "bar",
) -> Optional[dict[str, Any]]:
    if not stats.get("available"):
        return None
    rows = [
        {"label": key, "value": value}
        for key, value in stats["counts"].items()
    ]
    return _chart(chart_type, title, rows, "label", "value")


def _table(title: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"title": title, "rows": rows}


def _distribution_table(stats: Mapping[str, Any], title: str) -> dict[str, Any]:
    return _table(title, [
        {"label": label, "count": stats["counts"][label],
         "percentage": f"{stats['percentages'][label]:.2f}%"}
        for label in stats["counts"]
    ])


def _histogram_chart(values: pd.Series, title: str) -> dict[str, Any]:
    """Pre-binned histogram so no raw rows are copied into the chat history."""

    bins = int(min(10, max(3, values.nunique())))
    counts, edges = np.histogram(values, bins=bins)
    rows = [
        {"label": f"{edges[i]:.3g}–{edges[i + 1]:.3g}", "value": int(counts[i])}
        for i in range(len(counts))
    ]
    return _chart("bar", title, rows, "label", "value")


def _numeric_statistics(context: Mapping[str, Any]) -> dict[str, Any]:
    data = _context_data(context)
    values: dict[str, Any] = {}
    for column in context.get("numeric_columns", []):
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        if series.empty:
            continue
        values[str(column)] = {
            "count": int(series.size),
            "mean": round(float(series.mean()), 4),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "median": float(series.median()),
        }
    return values


def _date_trend(context: Mapping[str, Any]) -> dict[str, Any]:
    """Monthly trend: sentiment shares when predictions align, else counts."""

    date_columns = list(context.get("date_columns") or [])
    if not date_columns:
        return {"available": False, "reason": "No valid date column was found."}
    data = _context_data(context)
    results = _results_dataframe(context)

    chosen, raw_dates, sentiment = None, None, None
    if results is not None and "predicted_sentiment" in results.columns:
        for column in date_columns:
            aligned = _aligned_column(context, column)
            if aligned is not None:
                chosen, raw_dates = column, aligned
                sentiment = results["predicted_sentiment"]
                break
    if chosen is None:
        chosen, raw_dates = date_columns[0], data[date_columns[0]]

    frame = pd.DataFrame({"date": _to_datetime(raw_dates)})
    if sentiment is not None:
        frame["sentiment"] = sentiment
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return {"available": False,
                "reason": f"No parseable dates were found in {chosen}."}
    frame["period"] = frame["date"].dt.to_period("M").astype(str)

    if sentiment is not None and frame["sentiment"].notna().any():
        counts = (
            frame.dropna(subset=["sentiment"])
            .groupby(["period", frame["sentiment"].astype(str)])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        shares = counts.div(counts.sum(axis=1), axis=0) * 100
        rows = [
            {"label": str(period), "value": round(float(value), 2),
             "series": str(label)}
            for period, row in shares.iterrows()
            for label, value in row.items()
        ]
        title = "Monthly predicted sentiment share (%)"
        has_sentiment = True
    else:
        counts = frame["period"].value_counts().sort_index()
        rows = [{"label": str(p), "value": int(c)} for p, c in counts.items()]
        title = f"Records per month: {chosen}"
        has_sentiment = False
    periods = sorted(frame["period"].unique())
    return {
        "available": True,
        "date_column": chosen,
        "rows": rows,
        "title": title,
        "has_sentiment": has_sentiment,
        "first_period": periods[0],
        "last_period": periods[-1],
        "periods": len(periods),
        "valid_rows": int(len(frame)),
    }


def generate_eda_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    """Generate verified EDA statistics and compact chart specifications."""

    data = _context_data(context)
    numeric_columns = list(context.get("numeric_columns", []))
    charts: list[dict[str, Any]] = []
    numeric = _numeric_statistics(context)

    for column in numeric_columns[:2]:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if not values.empty:
            charts.append(_histogram_chart(values, f"Distribution of {column}"))

    categorical: dict[str, Any] = {}
    for column in list(context.get("categorical_columns", []))[:5]:
        counts = data[column].dropna().astype(str).value_counts().head(10)
        if counts.empty:
            continue
        categorical[str(column)] = {str(k): int(v) for k, v in counts.items()}
        if len(categorical) <= 2:
            charts.append(_chart(
                "bar",
                f"Distribution of {column}",
                [{"label": str(k), "value": int(v)} for k, v in counts.items()],
                "label",
                "value",
            ))

    correlations: dict[str, Any] = {}
    if len(numeric_columns) >= 2:
        matrix = (
            data[numeric_columns].apply(pd.to_numeric, errors="coerce")
            .corr().round(4).fillna(0)
        )
        correlations = matrix.to_dict()
        charts.append({
            "type": "heatmap",
            "title": "Numeric correlations",
            "data": correlations,
            "columns": matrix.columns.tolist(),
        })

    trend = _date_trend(context)
    date_trends: list[dict[str, Any]] = []
    if trend.get("available"):
        date_trends = trend["rows"]
        charts.append(_chart("line", trend["title"], date_trends, "label", "value"))

    return {
        "numeric_statistics": numeric,
        "categorical_distributions": categorical,
        "date_columns": list(context.get("date_columns", [])),
        "date_trends": date_trends,
        "correlations": correlations,
        "charts": charts,
    }


def generate_quick_insights(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the most useful verified findings currently available."""

    bullets: list[str] = []
    evidence: dict[str, Any] = {}
    sentiment = calculate_sentiment_statistics(context)
    if sentiment.get("available"):
        evidence["sentiment"] = sentiment
        bullets.append(f"Predicted sentiment: {_format_percentages(sentiment)}.")

    emotion = calculate_emotion_statistics(context)
    if emotion.get("available"):
        evidence["emotion"] = emotion
        common = max(emotion["counts"], key=emotion["counts"].get)
        bullets.append(
            f"The most common predicted emotion is {common} "
            f"({emotion['percentages'][common]:.2f}%)."
        )

    topics = get_top_topics(context)
    if topics.get("available"):
        evidence["topics"] = topics
        labels = [
            ", ".join(item["keywords"][:3])
            for item in topics["topics"] if item["keywords"]
        ]
        if labels:
            bullets.append("Top discovered themes: " + "; ".join(labels) + ".")

    keywords = get_keyword_statistics(context)
    if keywords.get("available") and keywords["keywords"]:
        evidence["keywords"] = keywords
        bullets.append(
            "Frequent feedback terms: "
            + ", ".join(item["keyword"] for item in keywords["keywords"][:5])
            + "."
        )

    ratings = _rating_columns(context)
    if ratings:
        column = ratings[0]
        values = pd.to_numeric(_context_data(context)[column], errors="coerce").dropna()
        if not values.empty:
            evidence["rating"] = {
                "column": column,
                "mean": round(float(values.mean()), 2),
                "median": round(float(values.median()), 2),
            }
            bullets.append(
                f"Average {column} is {values.mean():.2f} "
                f"with median {values.median():.2f}."
            )

    if not bullets:
        bullets.append(
            "Upload data and run at least one analysis to unlock model-based insights."
        )
    return {
        "available": bool(evidence),
        "headline": (
            "Here are the strongest findings currently available."
            if evidence else "No completed model findings are available yet."
        ),
        "bullets": bullets,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Answer handlers
# ---------------------------------------------------------------------------

Response = dict[str, Any]


def _new_response(question: str, intent: str) -> Response:
    return {
        "question": question,
        "intent": intent,
        "available": True,
        "answer": "",
        "evidence": {},
        "tables": [],
        "charts": [],
    }


def _unavailable(response: Response, reason: str) -> None:
    response["available"] = False
    response["answer"] = reason


def _h_unsupported(context: Mapping[str, Any], response: Response) -> None:
    response["available"] = False
    response["answer"] = (
        "I can answer questions about the dataset overview, sentiment, emotions, "
        "topics and keywords, customer complaints, ratings, and business "
        "insights. Please choose one of the suggested questions; I am not an "
        "open-ended conversational model."
    )


def _h_record_count(context: Mapping[str, Any], response: Response) -> None:
    data = _context_data(context)
    results = context.get("analysis_results") or {}
    text = f"This dataset contains {len(data):,} records (rows)."
    evidence: dict[str, Any] = {"records": int(len(data))}
    frame = _results_dataframe(context)
    if frame is not None:
        excluded = results.get("excluded_row_count")
        text += f" {len(frame):,} of them were analyzed"
        text += (f" and {int(excluded):,} were excluded." if excluded is not None else ".")
        evidence["analyzed_records"] = int(len(frame))
        if excluded is not None:
            evidence["excluded_records"] = int(excluded)
    response["answer"] = text
    response["evidence"] = evidence


def _column_kind(context: Mapping[str, Any], column: Any) -> str:
    if column in context.get("date_columns", []):
        return "date"
    if column in context.get("numeric_columns", []):
        return "numeric"
    if column == context.get("text_column"):
        return "feedback text"
    if column in context.get("categorical_columns", []):
        return "categorical"
    if column in context.get("text_columns", []):
        return "text"
    return "other"


def _h_columns(context: Mapping[str, Any], response: Response) -> None:
    data = _context_data(context)
    rows = [
        {"column": str(column), "type": _column_kind(context, column),
         "non_null": int(data[column].notna().sum())}
        for column in data.columns
    ]
    names = [str(column) for column in data.columns]
    shown = ", ".join(names[:30]) + (" …" if len(names) > 30 else "")
    response["answer"] = f"The dataset has {len(names):,} columns: {shown}."
    response["evidence"] = {"columns": names, "column_types": {
        row["column"]: row["type"] for row in rows}}
    response["tables"].append(_table("Columns and detected types", rows))


def _h_data_quality(context: Mapping[str, Any], response: Response) -> None:
    data = _context_data(context)
    issues: list[str] = []
    evidence: dict[str, Any] = {"rows": int(len(data))}

    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    total_missing = int(missing.sum())
    evidence["missing_values"] = total_missing
    rows = [
        {"column": str(column), "missing": int(count),
         "percentage": f"{count / len(data) * 100:.1f}%"}
        for column, count in missing.items()
    ]
    if total_missing:
        issues.append(
            f"{total_missing:,} missing values across {len(missing)} column(s), "
            f"most in {missing.index[0]} ({int(missing.iloc[0]):,})"
        )

    try:
        duplicates = int(data.duplicated().sum())
    except TypeError:
        duplicates = int(data.astype(str).duplicated().sum())
    evidence["duplicate_rows"] = duplicates
    if duplicates:
        issues.append(f"{duplicates:,} fully duplicated rows")

    empty_rows = int(data.isna().all(axis=1).sum())
    evidence["completely_empty_rows"] = empty_rows
    if empty_rows:
        issues.append(f"{empty_rows:,} completely empty rows")

    text_column = context.get("text_column")
    if text_column in data.columns:
        blank = int(
            data[text_column].dropna().astype(str).str.strip().eq("").sum()
        )
        unusable = blank + int(data[text_column].isna().sum())
        evidence["unusable_feedback_rows"] = unusable
        if unusable:
            issues.append(
                f"{unusable:,} rows have no usable text in the feedback "
                f"column “{text_column}”"
            )

    for column in context.get("date_columns", []):
        series = data[column].dropna()
        invalid = int(_to_datetime(series).isna().sum()) if len(series) else 0
        if invalid:
            evidence.setdefault("invalid_dates", {})[str(column)] = invalid
            issues.append(f"{invalid:,} unparseable dates in {column}")

    # A rating column holding an invalid value may no longer be detected as a
    # rating column, so columns that are merely *named* like ratings are checked too.
    rating_like = list(_rating_columns(context)) + [
        column for column in context.get("numeric_columns", [])
        if column not in _rating_columns(context)
        and any(key in str(column).lower() for key in ("rating", "score", "star"))
    ]
    for column in rating_like:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        outside = int(((values < 1) | (values > 5)).sum())
        if outside:
            evidence.setdefault("ratings_outside_1_5", {})[str(column)] = outside
            issues.append(f"{outside:,} values in {column} outside the 1–5 range")

    if issues:
        response["answer"] = (
            f"Checked {len(data):,} rows. Issues found: " + "; ".join(issues) + "."
        )
    else:
        response["answer"] = (
            f"No data-quality issues were found in {len(data):,} rows for the "
            "checks performed (missing values, duplicates, empty rows, unusable "
            "feedback text, invalid dates, and out-of-range ratings)."
        )
    response["evidence"] = evidence
    if rows:
        response["tables"].append(_table("Columns with missing values", rows))


def _h_eda(context: Mapping[str, Any], response: Response) -> None:
    summary = generate_eda_summary(context)
    charts = summary["charts"]
    response["evidence"] = {k: v for k, v in summary.items() if k != "charts"}
    response["charts"] = charts
    response["answer"] = (
        f"Basic EDA found {len(summary['numeric_statistics'])} numeric columns, "
        f"{len(summary['categorical_distributions'])} categorical columns, and "
        f"{len(summary['date_columns'])} date columns."
    )
    if summary["numeric_statistics"]:
        response["tables"].append(_table("Numeric summary", [
            {"column": column, **values}
            for column, values in summary["numeric_statistics"].items()
        ]))


def _h_numeric_stats(context: Mapping[str, Any], response: Response) -> None:
    stats = _numeric_statistics(context)
    if not stats:
        _unavailable(response, "No suitable numeric columns were found.")
        return
    response["evidence"] = stats
    preview = "; ".join(
        f"{column}: mean {values['mean']}, median {values['median']}, "
        f"min {values['minimum']:g}, max {values['maximum']:g}"
        for column, values in list(stats.items())[:3]
    )
    more = f" (+{len(stats) - 3} more in the table)" if len(stats) > 3 else ""
    response["answer"] = f"Numeric statistics — {preview}{more}."
    response["tables"].append(_table("Numeric statistics", [
        {"column": column, **values} for column, values in stats.items()
    ]))


def _h_date_trends(context: Mapping[str, Any], response: Response) -> None:
    trend = _date_trend(context)
    if not trend.get("available"):
        _unavailable(response, trend["reason"])
        return
    kind = "predicted sentiment share" if trend["has_sentiment"] else "record count"
    response["answer"] = (
        f"Monthly {kind} from {trend['first_period']} to {trend['last_period']} "
        f"({trend['periods']} months, {trend['valid_rows']:,} rows with valid "
        f"dates in {trend['date_column']})."
    )
    response["evidence"] = {
        "date_column": trend["date_column"],
        "date_trends": trend["rows"],
    }
    response["charts"].append(_chart("line", trend["title"], trend["rows"], "label", "value"))


def _h_sentiment(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_sentiment_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run sentiment analysis.")
        return
    response["answer"] = (
        f"Across {stats['total']:,} analyzed reviews, model predictions are: "
        f"{_format_percentages(stats)}."
    )
    response["tables"].append(_distribution_table(stats, "Predicted sentiment"))


def _h_sentiment_chart(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_sentiment_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run sentiment analysis.")
        return
    response["answer"] = f"Predicted sentiment distribution across {stats['total']:,} analyzed reviews."
    response["charts"].append(
        _distribution_chart(stats, "Predicted sentiment distribution", "pie"))
    response["tables"].append(_distribution_table(stats, "Predicted sentiment"))


def _h_sentiment_counts(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_sentiment_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run sentiment analysis.")
        return
    counts = stats["counts"]
    positive = sum(v for k, v in counts.items() if _is_label(k, "positive"))
    negative = sum(v for k, v in counts.items() if _is_label(k, "negative"))
    total = stats["total"]
    other = {k: v for k, v in counts.items()
             if not _is_label(k, "positive") and not _is_label(k, "negative")}
    text = (
        f"Of {total:,} analyzed reviews, {positive:,} "
        f"({positive / total * 100:.2f}%) were predicted positive and "
        f"{negative:,} ({negative / total * 100:.2f}%) negative"
    )
    if other:
        text += "; others: " + ", ".join(f"{k} {v:,}" for k, v in other.items())
    response["answer"] = text + "."
    response["evidence"] = {**stats, "positive": positive, "negative": negative}
    response["tables"].append(_distribution_table(stats, "Predicted sentiment"))


def _h_emotion(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_emotion_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run emotion analysis.")
        return
    top = list(stats["counts"])[:3]
    response["answer"] = (
        "Most frequent predicted emotions: "
        + ", ".join(f"{label} ({stats['percentages'][label]:.2f}%)" for label in top)
        + f". Based on {stats['total']:,} analyzed reviews."
    )
    response["tables"].append(_distribution_table(stats, "Predicted emotions"))


def _h_emotion_dominant(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_emotion_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run emotion analysis.")
        return
    labels = list(stats["counts"])
    top = labels[0]
    tied = [l for l in labels if stats["counts"][l] == stats["counts"][top]]
    if len(tied) > 1:
        response["answer"] = (
            f"{', '.join(tied)} are tied as the most frequent emotions, each "
            f"{stats['percentages'][top]:.2f}% of {stats['total']:,} analyzed reviews."
        )
    else:
        response["answer"] = (
            f"The dominant predicted emotion is {top}: {stats['counts'][top]:,} of "
            f"{stats['total']:,} analyzed reviews ({stats['percentages'][top]:.2f}%)."
        )


def _h_emotion_chart(context: Mapping[str, Any], response: Response) -> None:
    stats = calculate_emotion_statistics(context)
    response["evidence"] = stats
    if not stats["available"]:
        _unavailable(response, stats["reason"] + " Open Analyze and run emotion analysis.")
        return
    response["answer"] = f"Predicted emotion distribution across {stats['total']:,} analyzed reviews."
    response["charts"].append(_distribution_chart(stats, "Predicted emotion distribution"))
    response["tables"].append(_distribution_table(stats, "Predicted emotions"))


def _review_answer(
    context: Mapping[str, Any],
    response: Response,
    polarity: str,
) -> None:
    reviews = _review_rows(context, polarity, 5)
    response["evidence"] = reviews
    if not reviews["available"]:
        _unavailable(response, reviews["reason"])
        return
    if not reviews["reviews"]:
        response["answer"] = f"No analyzed reviews were predicted as {polarity}."
        return
    order = " with the highest model confidence" if reviews["ranked_by_confidence"] else ""
    response["answer"] = (
        f"Showing {len(reviews['reviews'])} of {reviews['total']:,} reviews "
        f"predicted as {polarity}{order}."
    )
    response["tables"].append(_table(
        f"Reviews predicted as {polarity}",
        [{("review" if k == "text" else k): v for k, v in item.items()}
         for item in reviews["reviews"]],
    ))


def _h_negative_reviews(context: Mapping[str, Any], response: Response) -> None:
    _review_answer(context, response, "negative")


def _h_positive_reviews(context: Mapping[str, Any], response: Response) -> None:
    _review_answer(context, response, "positive")


def _h_topics(context: Mapping[str, Any], response: Response) -> None:
    topics = get_top_topics(context)
    response["evidence"] = topics
    if not topics["available"]:
        _unavailable(response, topics["reason"] + " Open Analyze and include topic discovery.")
        return
    response["answer"] = "Most common discovered topics: " + "; ".join(
        f"Topic {item['topic']} ({item['count']:,} records, {item['percentage']:.1f}%): "
        f"{', '.join(item['keywords']) or 'no keywords stored'}"
        for item in topics["topics"]
    )
    response["tables"].append(_table("Discovered topics", [
        {"topic": f"Topic {item['topic']}", "records": item["count"],
         "share": f"{item['percentage']:.1f}%", "keywords": ", ".join(item["keywords"])}
        for item in topics["topics"]
    ]))
    response["charts"].append(_chart(
        "horizontal_bar", "Discovered topic frequency",
        [{"label": f"Topic {item['topic']}", "value": item["count"]}
         for item in topics["topics"]],
        "label", "value",
    ))


def _h_keywords(context: Mapping[str, Any], response: Response) -> None:
    keywords = get_keyword_statistics(context)
    response["evidence"] = keywords
    if not keywords["available"]:
        _unavailable(response, keywords["reason"])
        return
    response["answer"] = "Most frequent meaningful keywords: " + ", ".join(
        f"{item['keyword']} ({item['count']})" for item in keywords["keywords"]
    )
    response["charts"].append(_chart(
        "horizontal_bar", "Most frequent keywords",
        [{"label": item["keyword"], "value": item["count"]}
         for item in reversed(keywords["keywords"])],
        "label", "value",
    ))


def _areas_table(areas: Mapping[str, Any], title: str) -> dict[str, Any]:
    rows = []
    for item in areas["areas"]:
        row = {"area": item["label"], "reviews": item["reviews"]}
        if areas["basis"] == "topic":
            row["records_in_area"] = item["records"]
            row["rate"] = f"{item['rate']:.1f}%"
            row["keywords"] = ", ".join(item["keywords"])
        else:
            row["share"] = f"{item['share']:.1f}%"
        rows.append(row)
    return _table(title, rows)


def _areas_chart(areas: Mapping[str, Any], title: str) -> dict[str, Any]:
    return _chart(
        "horizontal_bar", title,
        [{"label": item["label"], "value": item["reviews"]}
         for item in reversed(areas["areas"])],
        "label", "value",
    )


def _describe_area(item: Mapping[str, Any], basis: str) -> str:
    if basis == "topic":
        words = ", ".join(item["keywords"][:4]) or "no keywords stored"
        return f"{item['label']} ({words}): {item['reviews']:,} reviews"
    return f"“{item['label']}”: {item['reviews']:,} reviews"


def _h_complaints(context: Mapping[str, Any], response: Response) -> None:
    areas = _label_areas(context, "negative")
    response["evidence"] = areas
    if not areas["available"]:
        _unavailable(response, areas["reason"])
        return
    if not areas["areas"]:
        response["answer"] = "No analyzed reviews were predicted as negative, so there are no complaints to summarize."
        return
    basis = "discovered topics" if areas["basis"] == "topic" else "keywords"
    response["answer"] = (
        f"Among {areas['label_total']:,} negative reviews, the most frequent "
        f"complaint areas by {basis} are: "
        + "; ".join(_describe_area(i, areas["basis"]) for i in areas["areas"][:3]) + "."
    )
    if areas["basis"] == "keyword":
        response["answer"] += " Run topic discovery to group these into themes."
    response["tables"].append(_areas_table(areas, "Complaint areas"))
    response["charts"].append(_areas_chart(areas, "Negative reviews by area"))


def _h_recurring_issues(context: Mapping[str, Any], response: Response) -> None:
    frame, reason = _label_frame(context, "negative")
    if frame is None:
        _unavailable(response, reason)
        return
    total = int(len(frame))
    if total == 0:
        response["answer"] = "No analyzed reviews were predicted as negative."
        response["evidence"] = {"available": True, "negative_total": 0}
        return
    counts = [(w, c) for w, c in _keyword_counts(frame["text"], 15, per_review=True) if c >= 2]
    response["evidence"] = {"available": True, "negative_total": total,
                            "terms": [{"keyword": w, "reviews": c} for w, c in counts]}
    if not counts:
        response["answer"] = (
            f"No meaningful term appears in more than one of the {total:,} "
            "negative reviews, so no repeated issue stands out."
        )
        return
    response["answer"] = (
        f"Terms that recur across {total:,} negative reviews: "
        + ", ".join(f"{w} ({c} reviews)" for w, c in counts[:8]) + "."
    )
    response["tables"].append(_table("Terms shared by negative reviews", [
        {"term": w, "reviews": c, "share": f"{c / total * 100:.1f}%"}
        for w, c in counts
    ]))
    response["charts"].append(_chart(
        "horizontal_bar", "Recurring terms in negative reviews",
        [{"label": w, "value": c} for w, c in reversed(counts[:10])],
        "label", "value",
    ))


def _h_improvement(context: Mapping[str, Any], response: Response) -> None:
    areas = _label_areas(context, "negative")
    response["evidence"] = areas
    if not areas["available"]:
        _unavailable(response, areas["reason"])
        return
    if not areas["areas"]:
        response["answer"] = "No analyzed reviews were predicted as negative, so no improvement areas are indicated."
        return
    lines = "; ".join(
        f"{i}. {_describe_area(item, areas['basis'])}"
        for i, item in enumerate(areas["areas"][:3], start=1)
    )
    response["answer"] = (
        f"Ranked by number of negative reviews (of {areas['label_total']:,}): {lines}."
    )
    if areas["basis"] == "topic":
        highest = max(areas["areas"], key=lambda item: item["rate"])
        response["answer"] += (
            f" {highest['label']} has the highest negative rate "
            f"({highest['rate']:.1f}% of its {highest['records']:,} records)."
        )
    response["tables"].append(_areas_table(areas, "Improvement areas"))
    response["charts"].append(_areas_chart(areas, "Negative reviews by area"))


def _h_focus_first(context: Mapping[str, Any], response: Response) -> None:
    areas = _label_areas(context, "negative")
    response["evidence"] = areas
    if not areas["available"]:
        _unavailable(response, areas["reason"])
        return
    if not areas["areas"]:
        response["answer"] = "No analyzed reviews were predicted as negative, so no priority area is indicated."
        return
    first = areas["areas"][0]
    tied = [a["label"] for a in areas["areas"] if a["reviews"] == first["reviews"]]
    response["answer"] = (
        f"Start with {_describe_area(first, areas['basis'])} — the area mentioned "
        f"in the most negative reviews ({first['reviews'] / areas['label_total'] * 100:.1f}% "
        f"of {areas['label_total']:,})."
    )
    if len(tied) > 1:
        response["answer"] += " Tied with: " + ", ".join(t for t in tied if t != first["label"]) + "."
    response["tables"].append(_areas_table(areas, "Priority areas"))


def _h_positive_areas(context: Mapping[str, Any], response: Response) -> None:
    areas = _label_areas(context, "positive")
    response["evidence"] = areas
    if not areas["available"]:
        _unavailable(response, areas["reason"])
        return
    if not areas["areas"]:
        response["answer"] = "No analyzed reviews were predicted as positive."
        return
    response["answer"] = (
        f"Among {areas['label_total']:,} positive reviews, the strongest feedback areas are: "
        + "; ".join(_describe_area(i, areas["basis"]) for i in areas["areas"][:3]) + "."
    )
    response["tables"].append(_areas_table(areas, "Positive feedback areas"))
    response["charts"].append(_areas_chart(areas, "Positive reviews by area"))


def _h_ratings(context: Mapping[str, Any], response: Response) -> None:
    columns = _rating_columns(context)
    if not columns:
        _unavailable(response, "No valid numeric rating column was found in this dataset.")
        return
    data = _context_data(context)
    column = columns[0]
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        _unavailable(response, f"The rating column “{column}” has no numeric values.")
        return
    distribution = values.value_counts().sort_index()
    low = int((values <= 2).sum())
    high = int((values >= 4).sum())
    evidence: dict[str, Any] = {
        "column": column,
        "count": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "low_share": round(low / values.size * 100, 2),
        "high_share": round(high / values.size * 100, 2),
    }
    text = (
        f"{column} (from {values.size:,} rated records): mean {values.mean():.2f}, "
        f"median {values.median():g}, range {values.min():g}–{values.max():g}. "
        f"{evidence['high_share']:.1f}% are 4 or higher and "
        f"{evidence['low_share']:.1f}% are 2 or lower."
    )
    sentiment = _results_dataframe(context)
    aligned = _aligned_column(context, column)
    if (
        sentiment is not None and aligned is not None
        and "predicted_sentiment" in sentiment.columns
    ):
        frame = pd.DataFrame({
            "sentiment": sentiment["predicted_sentiment"].astype(str),
            "rating": pd.to_numeric(aligned, errors="coerce"),
        }).dropna()
        if not frame.empty:
            grouped = frame.groupby("sentiment")["rating"].agg(["mean", "count"])
            response["tables"].append(_table(
                f"Average {column} by predicted sentiment",
                [{"sentiment": label, "average_rating": round(float(row["mean"]), 2),
                  "reviews": int(row["count"])} for label, row in grouped.iterrows()],
            ))
    response["answer"] = text
    response["evidence"] = evidence
    response["charts"].append(_chart(
        "bar", f"Rating distribution: {column}",
        [{"label": f"{k:g}", "value": int(v)} for k, v in distribution.items()],
        "label", "value",
    ))


def _h_category_compare(context: Mapping[str, Any], response: Response) -> None:
    results = _results_dataframe(context)
    if results is None or "predicted_sentiment" not in results.columns:
        _unavailable(response, "Run sentiment analysis first.")
        return
    for column in context.get("categorical_columns", []):
        aligned = _aligned_column(context, column)
        if aligned is None:
            continue
        frame = pd.DataFrame({
            "group": aligned.astype(str),
            "sentiment": results["predicted_sentiment"].astype(str),
        })
        grouped = (
            frame.groupby("group")["sentiment"]
            .value_counts(normalize=True).mul(100).rename("percentage").reset_index()
        )
        response["evidence"] = {"column": column,
                                "values": grouped.to_dict("records")}
        response["answer"] = f"Sentiment percentages grouped by {column} are shown in the table."
        response["tables"].append(_table(f"Sentiment by {column}", [
            {column: r["group"], "sentiment": r["sentiment"],
             "percentage": round(float(r["percentage"]), 2)}
            for r in grouped.to_dict("records")
        ]))
        return
    _unavailable(response, "No categorical column could be matched to the analyzed rows.")


def _h_business_insights(context: Mapping[str, Any], response: Response) -> None:
    insights = generate_quick_insights(context)
    response["available"] = insights["available"]
    response["evidence"] = insights["evidence"]
    response["answer"] = insights["headline"] + " " + " ".join(insights["bullets"])


_HANDLERS: dict[str, Callable[[Mapping[str, Any], Response], None]] = {
    "record_count": _h_record_count,
    "overview": _h_record_count,
    "columns": _h_columns,
    "data_quality": _h_data_quality,
    "eda": _h_eda,
    "numeric_stats": _h_numeric_stats,
    "date_trends": _h_date_trends,
    "sentiment": _h_sentiment,
    "sentiment_chart": _h_sentiment_chart,
    "sentiment_counts": _h_sentiment_counts,
    "emotion": _h_emotion,
    "emotion_dominant": _h_emotion_dominant,
    "emotion_chart": _h_emotion_chart,
    "negative_reviews": _h_negative_reviews,
    "positive_reviews": _h_positive_reviews,
    "topics": _h_topics,
    "keywords": _h_keywords,
    "complaints": _h_complaints,
    "recurring_issues": _h_recurring_issues,
    "improvement": _h_improvement,
    "focus_first": _h_focus_first,
    "positive_areas": _h_positive_areas,
    "rating_stats": _h_ratings,
    "rating_insights": _h_ratings,
    "category_compare": _h_category_compare,
    "business_insights": _h_business_insights,
    "unsupported": _h_unsupported,
}


def answer_question(
    question: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Answer a supported question using only supplied data and results."""

    intent = detect_intent(question)
    response = _new_response(question, intent)
    handler = _HANDLERS.get(intent, _h_unsupported)
    try:
        handler(context, response)
    except (KeyError, ValueError, TypeError, IndexError) as error:
        response.update({
            "available": False,
            "answer": (
                "I could not compute that from the current data "
                f"({type(error).__name__}: {error})."
            ),
            "evidence": {},
            "tables": [],
            "charts": [],
        })
    return response