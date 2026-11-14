
import pandas as pd

from .data_profiler import (
    profile_dataset,
    detect_text_columns,
    detect_label_columns
)
from .preprocessing import prepare_text_column
from .sentiment import predict_sentiment
from .emotion import predict_emotions
from .topics import discover_topics
from .evaluation import evaluate_classification


def normalize_sentiment_label(value):
    """Convert supported ground-truth labels to binary sentiment."""

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip().lower()

    label_map = {
        0: "negative",
        1: "positive",
        0.0: "negative",
        1.0: "positive",
        "0": "negative",
        "1": "positive",
        "negative": "negative",
        "positive": "positive"
    }

    return label_map.get(value)


def normalize_analysis_name(value):
    """Normalize supported analysis names to backend identifiers."""

    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ").replace("/", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "sentiment analysis": "sentiment",
        "emotion detection": "emotion",
        "topic discovery": "topics",
        "topic theme discovery": "topics",
        "model evaluation": "evaluation"
    }

    return aliases.get(normalized, normalized)


def select_automatic_analyses(data, label_column=None):
    """
    Select suitable analyses based on available dataset columns.

    Evaluation is selected only when compatible binary
    sentiment labels are available.
    """

    selected = ["sentiment", "emotion", "topics"]

    if label_column and label_column in data.columns:

        normalized_labels = {
            normalize_sentiment_label(value)
            for value in data[label_column].dropna()
        }

        normalized_labels.discard(None)

        if normalized_labels == {"negative", "positive"}:
            selected.append("evaluation")

    return selected



def run_automatic_analysis(
    data,
    text_column=None,
    label_column=None,
    sentiment_model="tfidf"
):
    """
    Automatically detect columns and select suitable analyses.
    Supports profiler results returned as strings or dictionaries.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The dataset is empty.")

    detected_text = detect_text_columns(data)
    detected_labels = detect_label_columns(data)

    # Normalize detected column names.
    def get_column_name(item):
        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            return item.get("column") or item.get("name")

        return None

    detected_text_names = [
        name for name in
        (get_column_name(item) for item in detected_text)
        if name
    ]

    detected_label_names = [
        name for name in
        (get_column_name(item) for item in detected_labels)
        if name
    ]

    # Select text column.
    if text_column is None:
        if not detected_text_names:
            raise ValueError(
                "No likely text column was detected. "
                "Please select one manually."
            )

        text_column = detected_text_names[0]

    if text_column not in data.columns:
        raise ValueError(
            f"Text column '{text_column}' not found."
        )

    # Select label column if available.
    if label_column is None and detected_label_names:
        label_column = detected_label_names[0]

    if label_column is not None and label_column not in data.columns:
        label_column = None

    # Select analyses.
    selected_analyses = select_automatic_analyses(
        data,
        label_column=label_column
    )

    # Run analysis pipeline.
    results = run_analysis(
        data=data,
        text_column=text_column,
        analyses=selected_analyses,
        label_column=label_column,
        sentiment_model=sentiment_model
    )

    results["mode"] = "automatic"
    results["selected_analyses"] = selected_analyses

    results["automatic_selection"] = {
        "text_column": text_column,
        "label_column": label_column,
        "detected_text_columns": detected_text_names,
        "detected_label_columns": detected_label_names
    }

    return results



def run_analysis(
    data,
    text_column,
    analyses=None,
    label_column=None,
    sentiment_model="tfidf"
):
    """
    Run selected analyses on a feedback dataset.

    Available analyses:
    sentiment, emotion, topics, evaluation
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The dataset is empty.")

    if text_column not in data.columns:
        raise ValueError(
            f"Text column '{text_column}' not found."
        )

    if analyses is None:
        analyses = ["sentiment", "topics"]

    if isinstance(analyses, str):
        analyses = [analyses]

    analyses = [
        normalize_analysis_name(analysis)
        for analysis in analyses
    ]

    # Preserve the caller's order while ensuring each analysis runs once.
    analyses = list(dict.fromkeys(analyses))

    valid_analyses = {
        "sentiment",
        "emotion",
        "topics",
        "evaluation"
    }

    unknown = set(analyses) - valid_analyses

    if unknown:
        raise ValueError(
            f"Unsupported analyses: {sorted(unknown)}"
        )

    results = {
        "dataset_profile": profile_dataset(data),
        "text_column": text_column,
        "detected_text_columns": detect_text_columns(data),
        "detected_label_columns": detect_label_columns(data),
        "selected_analyses": analyses,
        "analyses": {}
    }

    # Prepare and filter feedback text.
    original_text = data[text_column]
    missing_feedback_mask = original_text.isna()

    prepared_data = prepare_text_column(data, text_column)
    cleaned_text = prepared_data[text_column]
    blank_feedback_mask = (
        original_text.notna()
        & cleaned_text.astype("string").str.strip().eq("")
    )

    valid_mask = (
        prepared_data[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    working_data = prepared_data.loc[valid_mask].copy()

    texts = working_data[text_column].astype(str).tolist()

    if not texts:
        raise ValueError(
            "No non-empty feedback text was found."
        )

    exclusion_reasons = []

    for position, row_index in enumerate(data.index):
        if valid_mask.iloc[position]:
            continue

        reasons = []

        if missing_feedback_mask.iloc[position]:
            reasons.append("missing feedback value")

        if blank_feedback_mask.iloc[position]:
            reasons.append("blank or whitespace-only feedback")

        if not reasons:
            reasons.append("no usable feedback after cleaning")

        exclusion_reasons.append({
            "row_index": row_index,
            "reasons": reasons
        })

    # SENTIMENT ANALYSIS
    if "sentiment" in analyses:

        predictions = predict_sentiment(
            texts,
            model_type=sentiment_model
        )

        if len(predictions) != len(working_data):
            raise ValueError(
                "Sentiment prediction count does not match input rows."
            )

        working_data["predicted_sentiment"] = [
            item["predicted_sentiment"]
            for item in predictions
        ]

        working_data["sentiment_confidence"] = [
            item["confidence"]
            for item in predictions
        ]

        results["analyses"]["sentiment"] = {
            "status": "completed",
            "predictions": predictions,
            "distribution": working_data[
                "predicted_sentiment"
            ].value_counts().to_dict()
        }

    # EMOTION ANALYSIS
    if "emotion" in analyses:

        predictions = predict_emotions(texts)

        if len(predictions) != len(working_data):
            raise ValueError(
                "Emotion prediction count does not match input rows."
            )

        working_data["predicted_emotion"] = [
            item["predicted_emotion"]
            for item in predictions
        ]

        working_data["emotion_confidence"] = [
            item["confidence"]
            for item in predictions
        ]

        results["analyses"]["emotion"] = {
            "status": "completed",
            "predictions": predictions,
            "distribution": working_data[
                "predicted_emotion"
            ].value_counts().to_dict()
        }

    # TOPIC MODELLING
    if "topics" in analyses:

        try:
            topic_results = discover_topics(texts)

            assignments = topic_results["document_topic"]

            if len(assignments) != len(working_data):
                raise ValueError(
                    "Topic assignment count does not match input rows."
                )

            working_data["discovered_topic"] = assignments

            results["analyses"]["topics"] = {
                "status": "completed",
                "topics": topic_results["topics"],
                "document_topic": assignments
            }

        except ValueError as error:

            results["analyses"]["topics"] = {
                "status": "skipped",
                "reason": str(error)
            }

    # EVALUATION
    if "evaluation" in analyses:

        if (
            label_column is None
            or label_column not in working_data.columns
        ):

            results["analyses"]["evaluation"] = {
                "status": "skipped",
                "reason": (
                    "A valid ground-truth label column "
                    "was not supplied."
                )
            }

        elif "sentiment" not in analyses:

            results["analyses"]["evaluation"] = {
                "status": "skipped",
                "reason": (
                    "Run sentiment analysis before evaluating predictions."
                )
            }

        else:

            actual_raw = working_data[label_column].tolist()

            actual_normalized = [
                normalize_sentiment_label(value)
                for value in actual_raw
            ]

            predicted = (
                working_data["predicted_sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
                .tolist()
            )

            valid_evaluation_mask = [
                actual is not None
                and prediction in {"positive", "negative"}
                for actual, prediction in zip(
                    actual_normalized,
                    predicted
                )
            ]

            valid_actual = [
                actual
                for actual, valid in zip(
                    actual_normalized,
                    valid_evaluation_mask
                )
                if valid
            ]

            valid_predicted = [
                prediction
                for prediction, valid in zip(
                    predicted,
                    valid_evaluation_mask
                )
                if valid
            ]

            excluded_count = (
                len(working_data) - len(valid_actual)
            )

            if (
                len(valid_actual) == 0
                or len(set(valid_actual)) < 2
            ):

                results["analyses"]["evaluation"] = {
                    "status": "skipped",
                    "reason": (
                        "At least two ground-truth sentiment classes "
                        "are required for evaluation."
                    ),
                    "evaluated_rows": len(valid_actual),
                    "excluded_rows": excluded_count
                }

            else:

                try:
                    metrics = evaluate_classification(
                        valid_actual,
                        valid_predicted
                    )

                    results["analyses"]["evaluation"] = {
                        "status": (
                            "completed_with_exclusions"
                            if excluded_count > 0
                            else "completed"
                        ),
                        "evaluated_rows": len(valid_actual),
                        "excluded_rows": excluded_count,
                        "metrics": metrics
                    }

                except ValueError as error:

                    results["analyses"]["evaluation"] = {
                        "status": "skipped",
                        "reason": str(error),
                        "evaluated_rows": len(valid_actual),
                        "excluded_rows": excluded_count
                    }

    # FINAL RESULTS
    analyzed_row_count = len(working_data)
    excluded_row_count = len(data) - analyzed_row_count

    results["row_count_analyzed"] = len(working_data)
    results["row_count_original"] = len(data)
    results["row_count_excluded"] = excluded_row_count
    results["original_row_count"] = len(data)
    results["excluded_row_count"] = excluded_row_count
    results["row_accounting"] = {
        "original_rows": len(data),
        "completely_empty_rows": results[
            "dataset_profile"
        ]["completely_empty_rows"],
        "rows_with_missing_feedback": int(
            missing_feedback_mask.sum()
        ),
        "rows_with_blank_feedback": int(
            blank_feedback_mask.sum()
        ),
        "rows_analyzed": analyzed_row_count,
        "rows_excluded": excluded_row_count,
        "exclusion_reasons": exclusion_reasons
    }

    results["results_dataframe"] = working_data

    results["analysis_summary"] = {
        analysis_name: analysis_result.get(
            "status",
            "completed"
        )
        for analysis_name, analysis_result
        in results["analyses"].items()
    }

    return results
