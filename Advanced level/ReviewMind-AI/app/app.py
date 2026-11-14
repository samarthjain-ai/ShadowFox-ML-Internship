"""ReviewMind AI Streamlit application."""

import html
import re
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant_engine import (
    answer_question,
    build_dataset_signature,
    get_dataset_context,
    get_demo_dataset,
    get_guided_questions,
    get_keyword_statistics,
    reset_assistant_context,
)
from src.analysis_engine import run_analysis, run_automatic_analysis
from src.data_loader import inspect_workbook, load_dataset
from src.data_profiler import (
    detect_label_columns,
    detect_text_columns,
    profile_dataset,
    validate_feedback_column,
)
from src.exporter import build_report_pdf
from src.visualization import (
    DATE_GROUPINGS,
    MAX_PAIR_COLUMNS,
    SORT_ORDERS,
    allowed_aggregations,
    available_chart_types,
    build_chart,
    build_custom_chart,
    chart_bytes,
    classify_columns,
    column_options,
    column_type_table,
    default_date_grouping,
    get_chart_spec,
    supported_charts,
)


st.set_page_config(page_title="ReviewMind AI", page_icon="🧠", layout="wide")

with open(Path(__file__).parent / "style.css", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit version compatibility
# ---------------------------------------------------------------------------

_ST_VERSION = tuple(int(part) for part in re.findall(r"\d+", st.__version__)[:2])

# ``use_container_width`` is deprecated in newer Streamlit releases in favour
# of ``width="stretch"``. Older releases only understand the former.
WIDE = {"width": "stretch"} if _ST_VERSION >= (1, 51) else {"use_container_width": True}

_fragment_decorator = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def fragment(function):
    """Run ``function`` as a Streamlit fragment when the version supports it."""

    return _fragment_decorator(function) if _fragment_decorator else function


def keyed_container(key):
    """A container with a stable ``st-key-<key>`` CSS class (Streamlit >= 1.39)."""

    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


TERMS = {
    "Sentiment": ("The overall positive or negative tone of feedback.",
                  "Example: “Fast delivery” is likely positive."),
    "Emotions": ("A finer-grained label such as joy, anger, sadness, or fear.",
                 "Emotion detection adds nuance beyond positive/negative."),
    "Confidence score": ("The model's estimated certainty for one prediction.",
                         "Lower confidence means the result deserves more review."),
    "TF-IDF": ("A text representation that gives weight to useful, distinctive terms.",
               "It helps Logistic Regression learn which words correlate with sentiment."),
    "Logistic Regression": ("A fast supervised classifier used here with TF-IDF features.",
                            "It predicts a class probability for each feedback record."),
    "DistilBERT": ("A compact transformer model that understands context in text.",
                   "It can capture phrasing that simple word counts may miss."),
    "Topic modelling": ("An unsupervised method for grouping recurring language into themes.",
                        "ReviewMind uses TF-IDF and NMF to discover topic keywords."),
    "Evaluation metrics": ("Accuracy, precision, recall, F1, and ROC-AUC compare predictions with labels.",
                           "Metrics are shown only when a valid binary label column exists."),
}

# Analysis widgets whose values must survive visiting other pages. Streamlit
# discards the state of widgets that are not drawn in a run, which would
# otherwise reset the configuration (and discard the results) on navigation.
# Custom Graph Builder input widgets all use the ``custom_graph_`` key prefix and
# are kept automatically by ``keep_widget_state``. Its buttons use ``graph_builder_``
# keys because Streamlit does not allow button values to be assigned.
PERSISTED_WIDGET_KEYS = (
    "selected_text", "selected_label", "analysis_mode", "analysis_model",
    "analysis_sentiment", "analysis_emotion", "analysis_topics", "analysis_evaluation",
)


def keep_widget_state():
    for key in PERSISTED_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    for key in [name for name in st.session_state.keys() if str(name).startswith("custom_graph_")]:
        st.session_state[key] = st.session_state[key]


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()


def show_empty_state():
    st.markdown(
        '<div class="hero"><span class="eyebrow">CUSTOMER FEEDBACK INTELLIGENCE</span>'
        '<h1>Turn feedback into decisions.</h1>'
        '<p>ReviewMind AI analyzes sentiment, emotions, topics, ratings, and trends '
        'locally — without paid APIs or fabricated answers.</p></div>',
        unsafe_allow_html=True,
    )
    one, two, three = st.columns(3)
    one.markdown("### 01 · Load data\nUpload CSV/Excel or start with the guided demo.")
    two.markdown("### 02 · Analyze\nChoose automatic or custom local ML analyses.")
    three.markdown("### 03 · Discover\nAsk grounded questions and export findings.")
    if st.button("Try the guided demo", type="primary"):
        st.session_state["demo_requested"] = True
        rerun()


def load_current_data():
    """Load a source once and keep it; reset stale state when the source changes.

    The parsed DataFrame is cached against a cheap source key so that clicking
    chat buttons or changing widgets never re-reads or re-hashes the file.
    ``original_data`` always holds an untouched copy of the loaded dataset.
    """

    using_demo = st.session_state.get("use_demo", False)
    uploaded = st.session_state.get("uploaded_file")
    if not using_demo and uploaded is None:
        return None

    if using_demo:
        name, sheet, header = "ReviewMind demo dataset", None, 0
        source = ("demo",)
        loader = None
    else:
        name = uploaded.name
        file_id = getattr(uploaded, "file_id", None) or f"{name}:{getattr(uploaded, 'size', 0)}"
        sheet, header = None, 0
        if Path(name).suffix.lower() in {".xlsx", ".xls"}:
            workbook = st.session_state.get("workbook_info")
            if not workbook or workbook["file_id"] != file_id:
                workbook = {"file_id": file_id, "info": inspect_workbook(uploaded, preview_rows=12)}
                st.session_state["workbook_info"] = workbook
            info = workbook["info"]
            sheet = st.selectbox("Worksheet", info["sheet_names"], key=f"selected_sheet_{file_id}")
            selected = next(item for item in info["sheets"] if item["name"] == sheet)
            suggested = selected["suggested_header_row"]
            header = st.selectbox(
                "Header row (0-based)", list(range(max(1, suggested + 2))),
                index=min(suggested, max(1, suggested + 2) - 1),
                key=f"selected_header_{file_id}_{sheet}",
            )
            with st.expander("Preview workbook rows"):
                st.dataframe(selected["preview"], **WIDE)
        source = (file_id, name, sheet, header)
        loader = (uploaded, sheet, header)

    cached = st.session_state.get("loaded_data_cache")
    if cached and cached["source"] == source:
        return cached["data"]

    if loader is None:
        data = get_demo_dataset()
        raw = data.to_csv(index=True).encode()
    else:
        file, sheet, header = loader
        if sheet is not None:
            data = load_dataset(file, sheet_name=sheet, header=header)
        else:
            data = load_dataset(file)
        raw = file.getvalue()

    signature = build_dataset_signature(raw, name, sheet, header)
    changed = reset_assistant_context(st.session_state, signature)
    if changed or not isinstance(st.session_state.get("original_data"), pd.DataFrame):
        st.session_state["original_data"] = data.copy()
        # Same object on purpose: the assistant reads it but never writes to it.
        st.session_state["assistant_dataset"] = st.session_state["original_data"]
        st.session_state["assistant_metadata"] = {
            "file_name": name, "sheet_name": sheet, "header": header,
        }
    st.session_state["loaded_data_cache"] = {"source": source, "data": data}
    return data


def show_overview(data):
    profile = profile_dataset(data)
    st.subheader("Dataset health")
    cols = st.columns(5)
    for column, label, value in zip(
        cols,
        ("Rows", "Columns", "Missing values", "Duplicate rows", "Empty rows"),
        (profile["rows"], profile["columns"], profile["missing_value_total"],
         profile["duplicate_rows"], profile["completely_empty_rows"]),
    ):
        column.metric(label, f"{value:,}")
    with st.expander("Inspect schema and quality details"):
        st.dataframe(pd.DataFrame({
            "dtype": profile["data_types"],
            "missing": profile["missing_values"],
            "blank": profile["empty_string_values"],
        }), **WIDE)
        st.caption("Original uploaded data stays unchanged; analysis results are stored separately.")


def show_analysis(data):
    show_overview(data)
    detected_text = detect_text_columns(data)
    recommended = [item["column"] for item in detected_text]
    st.subheader("Configure analysis")
    if not detected_text:
        st.warning("No high-confidence feedback column was detected. Select a non-numeric text column.")
    options = data.columns.tolist()
    selected_text = st.selectbox(
        "Feedback column", options,
        index=options.index(recommended[0]) if recommended else 0,
        format_func=lambda value: f"{value} (recommended)" if value in recommended else str(value),
        key="selected_text",
    )
    try:
        validation = validate_feedback_column(data, selected_text)
        st.caption(f"{validation['usable_rows']:,} usable feedback rows ({validation['coverage_percentage']:.1f}% coverage).")
    except ValueError as error:
        st.error(str(error))
        return
    labels = ["None"] + detect_label_columns(data)
    selected_label = st.selectbox("Ground-truth sentiment label (optional)", labels, key="selected_label")
    mode = st.radio("Analysis mode", ["Automatic", "Custom"], horizontal=True, key="analysis_mode")
    model = st.selectbox(
        "Sentiment model", ["tfidf", "distilbert"],
        format_func=lambda x: "TF-IDF + Logistic Regression" if x == "tfidf" else "DistilBERT",
        key="analysis_model",
    )
    analyses = ["sentiment", "emotion", "topics"]
    if mode == "Custom":
        analyses = []
        for key, label in (("sentiment", "Sentiment"), ("emotion", "Emotion"), ("topics", "Topics"), ("evaluation", "Evaluation")):
            if st.checkbox(label, value=key != "evaluation", key=f"analysis_{key}"):
                analyses.append(key)
    else:
        st.info("Automatic mode selects sentiment, emotions, topics, and evaluation when a valid label is available.")
    config_signature = (
        selected_text, selected_label, mode, model, tuple(analyses)
    )
    if st.session_state.get("analysis_config_signature") != config_signature:
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analysis_text_column", None)
        st.session_state["analysis_config_signature"] = config_signature
    if st.button("Run analysis", type="primary", **WIDE):
        if not analyses:
            st.error("Select at least one analysis.")
            return
        try:
            with st.spinner("Running local models; this may take a moment..."):
                results = (
                    run_automatic_analysis(data, selected_text, selected_label if selected_label != "None" else None, model)
                    if mode == "Automatic"
                    else run_analysis(data, selected_text, analyses, selected_label if selected_label != "None" else None, model)
                )
            st.session_state["analysis_results"] = results
            # The chat and visualizations must use the column the models actually read.
            st.session_state["analysis_text_column"] = selected_text
            st.success("Analysis complete.")
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            st.error(f"Analysis could not be completed: {error}")
    results = st.session_state.get("analysis_results")
    if results:
        show_results(results)


def show_results(results):
    frame = results.get("results_dataframe")
    if not isinstance(frame, pd.DataFrame):
        st.warning("No analyzed rows are available.")
        return
    st.subheader("Analysis results")
    cols = st.columns(4)
    cols[0].metric("Original rows", f"{results.get('original_row_count', len(frame)):,}")
    cols[1].metric("Analyzed rows", f"{len(frame):,}")
    cols[2].metric("Excluded rows", f"{results.get('excluded_row_count', 0):,}")
    cols[3].metric("Analyses", ", ".join(results.get("selected_analyses", [])))
    evaluation = results.get("analyses", {}).get("evaluation", {})
    if evaluation.get("metrics"):
        st.caption("Evaluation against the selected ground-truth labels")
        metric_cols = st.columns(len(evaluation["metrics"]))
        for column, (name, value) in zip(metric_cols, evaluation["metrics"].items()):
            column.metric(name.replace("_", " ").title(), f"{value:.3f}")
    st.dataframe(frame.head(500), **WIDE)
    with st.expander("Analysis status and limitations"):
        st.json(results.get("analysis_summary", {}))
        st.caption("Rows with missing or blank feedback are excluded from model predictions and remain accounted for.")
    st.download_button("Download predictions CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       "reviewmind_analysis_results.csv", "text/csv", **WIDE)


# ---------------------------------------------------------------------------
# Floating "Ask ReviewMind" assistant
# ---------------------------------------------------------------------------

CATEGORY_ICONS = {
    "Dataset Overview": "📋",
    "Sentiment Analysis": "😊",
    "Emotion Analysis": "🎭",
    "Topics and Keywords": "🏷️",
    "Customer Complaints": "⚠️",
    "Business Insights": "💡",
}
MAX_CHAT_CHARTS = 6


def _esc(value):
    return html.escape(str(value), quote=True)


def _feedback_column(data):
    """The feedback column the analyses used, else the best detected candidate."""

    for key in ("analysis_text_column", "selected_text"):
        column = st.session_state.get(key)
        if column in data.columns:
            return column
    candidates = detect_text_columns(data)
    return candidates[0]["column"] if candidates else None


def get_chat_state():
    """Assistant context and supported questions, rebuilt only when inputs change.

    Returns ``None`` when no dataset is loaded. Nothing here runs a model: it
    only reads the preserved DataFrame and the stored analysis results.
    """

    data = st.session_state.get("original_data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None
    results = st.session_state.get("analysis_results")
    text_column = _feedback_column(data)
    key = (st.session_state.get("dataset_signature"), text_column)
    cached = st.session_state.get("assistant_context_cache")
    if cached and cached["key"] == key and cached["results_ref"] is results:
        return cached
    context = get_dataset_context(
        data, results or {}, text_column, st.session_state.get("assistant_metadata"),
    )
    cached = {
        "key": key,
        "results_ref": results,
        "context": context,
        "questions": get_guided_questions(context),
    }
    st.session_state["assistant_context_cache"] = cached
    return cached


def _open_chat(is_open):
    st.session_state["chat_open"] = bool(is_open)


def _select_category(category):
    st.session_state["chat_category"] = category


def _ask_question(question):
    """Button callback: answer one suggested question and record it once."""

    state = get_chat_state()
    if state is None:
        return
    history = st.session_state.setdefault("assistant_history", [])
    history.append(answer_question(question, state["context"]))


def _clear_chat():
    """Forget the conversation only; the uploaded data and analyses stay."""

    st.session_state["assistant_history"] = []
    st.session_state["chat_category"] = None


def _render_assistant_chart(chart):
    """Render a chart specification returned by the local assistant."""

    chart_type = chart.get("type")
    rows = chart.get("data", [])
    figure, axis = plt.subplots(figsize=(4.9, 2.8))
    try:
        if chart_type == "heatmap":
            frame = pd.DataFrame(rows)
            if frame.empty:
                return
            sns.heatmap(frame, annot=True, cmap="Purples", fmt=".2f", ax=axis,
                        cbar=False, annot_kws={"size": 7})
            axis.tick_params(labelsize=7)
        else:
            chart_data = pd.DataFrame(rows)
            if chart_data.empty:
                return
            x_column = chart.get("x", "label")
            y_column = chart.get("y", "value")
            if chart_type == "pie":
                axis.pie(
                    chart_data[y_column],
                    labels=chart_data[x_column],
                    autopct="%1.1f%%",
                    textprops={"fontsize": 8},
                )
            elif chart_type == "histogram":
                axis.hist(chart_data[x_column], bins=min(12, max(3, len(chart_data))))
            elif chart_type == "horizontal_bar":
                axis.barh(chart_data[x_column], chart_data[y_column], color="#6d4fe6")
                axis.tick_params(labelsize=8)
            elif chart_type == "line":
                if "series" in chart_data.columns:
                    for series, values in chart_data.groupby("series"):
                        axis.plot(values[x_column], values[y_column], marker="o", label=series)
                    axis.legend(fontsize=7)
                else:
                    axis.plot(chart_data[x_column], chart_data[y_column], marker="o", color="#6d4fe6")
                axis.tick_params(axis="x", rotation=35, labelsize=7)
                axis.tick_params(axis="y", labelsize=8)
            elif chart_type == "box":
                axis.boxplot(chart_data[y_column].dropna())
            else:
                axis.bar(chart_data[x_column], chart_data[y_column], color="#6d4fe6")
                axis.tick_params(axis="x", rotation=35, labelsize=7)
                axis.tick_params(axis="y", labelsize=8)
            if chart_type != "pie":
                axis.set_ylabel("" if chart_type == "horizontal_bar" else str(y_column), fontsize=8)
                axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(chart.get("title", "ReviewMind chart"), fontsize=9)
        figure.tight_layout()
        st.pyplot(figure, **WIDE)
    finally:
        plt.close(figure)


def _render_table(table):
    rows = table.get("rows") or []
    if not rows:
        return
    st.markdown(f'<div class="rm-evidence-title">{_esc(table.get("title", ""))}</div>',
                unsafe_allow_html=True)
    if "review" in rows[0]:
        cards = []
        for row in rows:
            meta = f"Row {_esc(row['row_index'])}" if "row_index" in row else "Review"
            if "confidence" in row:
                meta += f" · confidence {_esc(row['confidence'])}"
            cards.append(
                f'<div class="rm-review"><span class="rm-review-meta">{meta}</span>'
                f'{_esc(row["review"])}</div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE)


def _render_message(role, text, unavailable=False):
    if role == "user":
        body = f'<div class="rm-msg rm-user"><div class="rm-bubble">{_esc(text)}</div></div>'
    else:
        extra = " rm-unavailable" if unavailable else ""
        body = (
            '<div class="rm-msg rm-assistant"><span class="rm-avatar" aria-hidden="true">RM</span>'
            f'<div class="rm-bubble{extra}">{_esc(text)}</div></div>'
        )
    st.markdown(body, unsafe_allow_html=True)


def _render_exchange(item):
    _render_message("user", item.get("question", ""))
    _render_message("assistant", item.get("answer", ""), not item.get("available", True))
    for chart in (item.get("charts") or [])[:MAX_CHAT_CHARTS]:
        _render_assistant_chart(chart)
    for table in item.get("tables") or []:
        _render_table(table)


def _welcome_text(state):
    if state is None:
        return ("Hi! I'm Ask ReviewMind. Load a dataset first — upload a file or "
                "start the demo — and I'll answer questions from it.")
    text = ("Hi! I'm Ask ReviewMind. Choose a topic below and I'll answer from "
            "your uploaded data and the analyses you've already run.")
    if "Sentiment Analysis" not in state["questions"]:
        text += (" Run an analysis on the Analyze page to unlock sentiment, "
                 "emotion, topic, and complaint questions.")
    return text


def _scroll_chat_to_latest():
    """Keep the newest message visible (plain DOM scroll; no external code)."""

    # Charts load after the first paint, so retry for a few seconds while the
    # content height settles.
    script = (
        "<script>(function(){var d=window.parent.document,n=0;"
        "var go=function(){var m=d.querySelector('.st-key-chat_messages');"
        "if(m){m.scrollTop=m.scrollHeight;}"
        "if(++n>=25){clearInterval(t);}};"
        "var t=setInterval(go,120);go();})();</script>"
    )
    with keyed_container("chat_scroll"):
        try:
            components.html(script, height=0)
        except Exception:  # scrolling is cosmetic; never break the chat over it
            pass


@fragment
def render_chat_widget():
    """Fixed bottom-right launcher and chat window.

    Opening, closing and asking happen in button callbacks inside a fragment,
    so only this widget reruns — the page behind it, any trained-model results,
    and the original DataFrame are left untouched.
    """

    if not st.session_state.get("chat_open", False):
        with keyed_container("chat_launcher"):
            st.button("💬 Ask ReviewMind", key="chat_open_btn",
                      on_click=_open_chat, args=(True,), help="Open the ReviewMind assistant")
        return

    state = get_chat_state()
    history = st.session_state.get("assistant_history", [])

    with keyed_container("chat_panel"):
        st.markdown(
            '<div class="rm-chat-header"><div class="rm-chat-logo" aria-hidden="true">🧠</div>'
            '<div class="rm-chat-titles"><div class="rm-chat-brand">ReviewMind AI</div>'
            '<div class="rm-chat-name">Ask ReviewMind</div>'
            '<div class="rm-chat-status"><span class="rm-dot"></span>'
            'Answers from your data · runs locally</div></div></div>',
            unsafe_allow_html=True,
        )
        st.button("Clear", key="chat_clear_btn", on_click=_clear_chat,
                  disabled=not history, help="Clear the conversation (your data is kept)")
        st.button("✕", key="chat_close_btn", on_click=_open_chat, args=(False,),
                  help="Close chat")

        with keyed_container("chat_messages"):
            with keyed_container("chat_thread"):
                _render_message("assistant", _welcome_text(state))
                for item in history:
                    _render_exchange(item)
            _scroll_chat_to_latest()

        with keyed_container("chat_choices"):
            if state is None:
                return
            questions = state["questions"]
            category = st.session_state.get("chat_category")
            if category not in questions:
                category = None
            if category is None:
                st.markdown('<div class="rm-choices-title">Choose a topic</div>',
                            unsafe_allow_html=True)
                with keyed_container("chat_categories"):
                    for index, name in enumerate(questions):
                        st.button(
                            f"{CATEGORY_ICONS.get(name, '•')} {name}",
                            key=f"chat_cat_{index}",
                            on_click=_select_category, args=(name,),
                        )
            else:
                st.button("← All topics", key="chat_back_btn",
                          on_click=_select_category, args=(None,))
                st.markdown(
                    f'<div class="rm-choices-title">{_esc(CATEGORY_ICONS.get(category, ""))} '
                    f'{_esc(category)}</div>', unsafe_allow_html=True)
                category_index = list(questions).index(category)
                with keyed_container("chat_questions"):
                    for index, item in enumerate(questions[category]):
                        st.button(
                            item["question"],
                            key=f"chat_q_{category_index}_{index}",
                            on_click=_ask_question, args=(item["question"],),
                            help=item["reason"], **WIDE,
                        )


# ---------------------------------------------------------------------------
# Visualization Center
# ---------------------------------------------------------------------------

def show_visualizations(data):
    st.header("Visualization Center")
    current_data = st.session_state.get("original_data")
    if not isinstance(current_data, pd.DataFrame):
        current_data = data
    with st.expander("Custom Graph Builder", expanded=True):
        show_custom_graph_builder(current_data)
    st.divider()
    results = st.session_state.get("analysis_results") or {}
    keyword_counts = {}
    selected_text = _feedback_column(data)
    if selected_text and not data.empty:
        context = get_dataset_context(data, results, selected_text)
        keyword_counts = {item["keyword"]: item["count"] for item in get_keyword_statistics(context).get("keywords", [])}
    results = {**results, "keyword_counts": keyword_counts}
    try:
        names = supported_charts(data, results)
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Could not determine the supported visualizations: {error}")
        return
    labels = {"sentiment": "Sentiment donut", "emotion": "Emotion distribution", "topics": "Topic frequency",
              "rating": "Rating distribution", "keywords": "Keyword frequency", "trend": "Date trend",
              "numeric": "Numeric distribution", "correlation": "Correlation heatmap"}
    if not names:
        st.info("Run an analysis or upload numeric/date/rating data to unlock supported visualizations.")
        return
    selected = st.selectbox(
        "Visualization", names,
        format_func=lambda x: labels.get(x, str(x).replace("_", " ").title()),
        key="selected_visualization",
    )
    try:
        built = build_chart(data, selected, results)
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"This visualization could not be built from the current data: {error}")
        return
    if built is None:
        st.info("This visualization is not supported by the current data.")
        return
    figure, interpretation = built
    st.pyplot(figure, **WIDE)
    st.caption(interpretation)
    png = chart_bytes(figure)
    plt.close(figure)
    st.download_button("Download chart PNG", png, f"reviewmind_{selected}.png", "image/png")


# ---------------------------------------------------------------------------
# Custom Graph Builder
# ---------------------------------------------------------------------------

CUSTOM_GRAPH_PREFIX = "custom_graph_"
NO_VALUE = "None (count records)"
NO_GROUP = "None (no grouping)"
NO_COLOR = "None"


def _custom_graph_key(chart_type, name):
    slug = re.sub(r"[^a-z0-9]+", "_", chart_type.lower()).strip("_")
    return f"{CUSTOM_GRAPH_PREFIX}{slug}_{name}"


def _keep_valid_choice(key, options):
    """Drop a remembered selection that is no longer one of the options."""

    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _keep_valid_choices(key, options):
    if key in st.session_state:
        kept = [item for item in st.session_state[key] if item in options]
        if kept != list(st.session_state[key]):
            st.session_state[key] = kept


def _custom_graph_classes(data):
    """Column types for the current dataset; all builder state resets when it changes."""

    signature = (st.session_state.get("dataset_signature"), data.shape,
                 tuple(str(name) for name in data.columns))
    if st.session_state.get("custom_graph_dataset_key") != signature:
        for stale in [key for key in st.session_state.keys()
                      if str(key).startswith(CUSTOM_GRAPH_PREFIX)]:
            del st.session_state[stale]
        st.session_state["custom_graph_dataset_key"] = signature
    classes = st.session_state.get("custom_graph_classes")
    if classes is None:
        classes = classify_columns(data)
        st.session_state["custom_graph_classes"] = classes
    return classes


def show_custom_graph_builder(data):
    """Render the custom graph builder using the actual uploaded dataframe."""

    st.caption(
        "Pick a chart type, choose compatible columns, then press Generate Chart. "
        "Charts use only the current uploaded dataset — nothing is replaced with sample values."
    )
    if not isinstance(data, pd.DataFrame) or data.empty or data.shape[1] == 0:
        st.info("The current dataset has no rows to chart. Upload a dataset that contains data.")
        return
    try:
        classes = _custom_graph_classes(data)
        available, unavailable = available_chart_types(classes)
    except Exception as error:
        st.error(f"Could not inspect the dataset's columns: {error}")
        return
    if st.checkbox("Show how each column was detected", key="custom_graph_show_types"):
        st.dataframe(column_type_table(data, classes), hide_index=True, **WIDE)
    if not available:
        st.info(
            "No chart type fits this dataset yet: it needs at least one numeric or "
            "categorical column. Text and identifier-like columns are not charted."
        )
        return

    _keep_valid_choice("custom_graph_chart_type", available)
    chart_type = st.selectbox("Chart type", available, key="custom_graph_chart_type")
    spec = get_chart_spec(chart_type)
    st.caption(f"{spec['family']} chart · {spec['help']}")
    if unavailable:
        st.caption("Not offered for this dataset: " + "; ".join(
            f"{name} ({reason})" for name, reason in unavailable.items()))

    options = column_options(classes, chart_type)

    def key(name):
        return _custom_graph_key(chart_type, name)

    x_column = y_column = color_column = size_column = columns = None

    slots = [slot for slot in ("x", "y", "color", "size")
             if spec[slot] and not (slot == "color" and spec["multi"])]
    if slots:
        slot_columns = dict(zip(slots, st.columns(len(slots))))
        if "x" in slots:
            choices = options["x"]
            if spec["x"] == "category_optional":
                choices = [NO_GROUP] + choices
            _keep_valid_choice(key("x"), choices)
            picked = slot_columns["x"].selectbox(spec["x_text"], choices, key=key("x"))
            x_column = None if picked == NO_GROUP else picked
        if "y" in slots:
            choices = [name for name in options["y"] if name != x_column or spec["x"] != "numeric"]
            if spec["y"] == "numeric_optional":
                choices = [NO_VALUE] + choices
            _keep_valid_choice(key("y"), choices)
            picked = slot_columns["y"].selectbox(spec["y_text"], choices, key=key("y"))
            y_column = None if picked == NO_VALUE else picked
        if "color" in slots:
            pool = [name for name in options["color"] if name not in (x_column, y_column)]
            choices = pool if spec["color"] == "required" else [NO_COLOR] + pool
            _keep_valid_choice(key("color"), choices)
            if choices:
                picked = slot_columns["color"].selectbox(spec["color_text"], choices, key=key("color"))
                color_column = None if picked == NO_COLOR else picked
        if "size" in slots:
            pool = [name for name in options["size"] if name not in (x_column, y_column)]
            _keep_valid_choice(key("size"), pool)
            if pool:
                size_column = slot_columns["size"].selectbox(
                    "Bubble size (numeric)", pool, key=key("size"))

    if spec["multi"]:
        pool = options["multi"]
        multi_key = key("columns")
        if multi_key not in st.session_state:
            st.session_state[multi_key] = pool[:4 if chart_type == "Pair plot" else 12]
        _keep_valid_choices(multi_key, pool)
        columns = st.multiselect(spec["x_text"], pool, key=multi_key)
        if chart_type == "Pair plot":
            st.caption(f"Choose 2–{MAX_PAIR_COLUMNS} columns. Large datasets are sampled "
                       "and the chart says so.")
            pool = [name for name in options["color"] if name not in (columns or [])]
            choices = [NO_COLOR] + pool
            _keep_valid_choice(key("color"), choices)
            picked = st.selectbox(spec["color_text"], choices, key=key("color"))
            color_column = None if picked == NO_COLOR else picked

    aggregation = "Count"
    date_grouping = "month"
    sort_order = SORT_ORDERS[0]
    top_n = None
    bins = 20
    is_date_axis = bool(spec["time"] and x_column in classes["datetime"])
    flags = [bool(spec["agg"]), is_date_axis, bool(spec["sort"]), bool(spec["top_n"]),
             bool(spec["bins"])]
    if any(flags):
        row = iter(st.columns(sum(flags)))
        if spec["agg"]:
            choices = allowed_aggregations(chart_type, y_column is not None)
            _keep_valid_choice(key("aggregation"), choices)
            aggregation = next(row).selectbox(
                "Aggregation", choices, key=key("aggregation"),
                help="Count needs no value column. Sum, Average, Median, Minimum and "
                     "Maximum need a numeric value column.")
        if is_date_axis:
            grouping_options = [name.title() for name in DATE_GROUPINGS]
            group_for = key("group_for")
            if st.session_state.get(group_for) != x_column:
                try:
                    default_group = str(default_date_grouping(data, x_column)).title()
                except Exception:
                    default_group = "Month"
                if default_group not in grouping_options:
                    default_group = "Month"
                st.session_state[key("date_grouping")] = default_group
                st.session_state[group_for] = x_column
            _keep_valid_choice(key("date_grouping"), grouping_options)
            date_grouping = next(row).selectbox(
                "Date grouping", grouping_options, key=key("date_grouping"),
                help="Month is chosen automatically when the dataset has many unique dates.",
            ).lower()
        if spec["sort"]:
            sort_order = next(row).selectbox("Sort", SORT_ORDERS, key=key("sort"))
        if spec["top_n"]:
            cell = next(row)
            if cell.checkbox("Show only the top N", key=key("limit")):
                if key("top_n") not in st.session_state:
                    st.session_state[key("top_n")] = 10
                top_n = int(cell.number_input("N", min_value=1, max_value=200, step=1,
                                              key=key("top_n")))
        if spec["bins"]:
            if key("bins") not in st.session_state:
                st.session_state[key("bins")] = 20
            bins = int(next(row).slider("Number of bins", 5, 100, key=key("bins")))

    labels = st.columns(3)
    title = labels[0].text_input("Chart title", key=key("title"), placeholder="Automatic")
    x_label = labels[1].text_input("X-axis label", key=key("x_label"), placeholder="Automatic")
    y_label = labels[2].text_input("Y-axis label", key=key("y_label"), placeholder="Automatic")

    config = {
        "chart_type": chart_type,
        "grouping_column": x_column,
        "value_column": y_column,
        "aggregation": aggregation,
        "date_grouping": date_grouping,
        "color_column": color_column,
        "size_column": size_column,
        "columns": list(columns) if columns else None,
        "sort_order": sort_order,
        "top_n": top_n,
        "bins": bins,
        "title": title.strip() or None,
        "x_label": x_label.strip() or None,
        "y_label": y_label.strip() or None,
    }

    blocked = None
    if spec["multi"]:
        count = len(columns or [])
        if count < 2:
            blocked = "Select at least two numeric columns."
        elif chart_type == "Pair plot" and count > MAX_PAIR_COLUMNS:
            blocked = f"A pair plot supports at most {MAX_PAIR_COLUMNS} columns."
    if blocked:
        st.warning(blocked)
    if st.button("Generate Chart", type="primary", key="graph_builder_generate",
                 disabled=bool(blocked)):
        try:
            with st.spinner("Building chart..."):
                figure, table = build_custom_chart(data, **config)
                png = chart_bytes(figure)
            st.session_state["custom_graph_result"] = {"config": config, "png": png, "table": table}
        except ValueError as error:
            st.session_state["custom_graph_result"] = {"config": config, "error": str(error)}
        except Exception as error:  # never let an unsuitable column crash the page
            st.session_state["custom_graph_result"] = {
                "config": config,
                "error": f"Something unexpected went wrong ({type(error).__name__}: {error}). "
                         "Try different columns or another chart type.",
            }
        finally:
            plt.close("all")

    result = st.session_state.get("custom_graph_result")
    if not isinstance(result, dict):
        return
    if result.get("config") != config:
        st.info("Settings changed — choose Generate Chart to update the graph.")
        return
    if "error" in result:
        st.error(f"The chart could not be built: {result['error']}")
        return

    table = result["table"]
    st.image(result["png"])
    summary = table.attrs.get("row_summary") or {}
    if summary:
        text = f"Used {summary['valid_rows']:,} of {summary['total_rows']:,} rows"
        if summary.get("excluded_rows"):
            text += (f" — {summary['excluded_rows']:,} rows were left out because a selected "
                     "column was missing or not usable")
        st.caption(text + ".")
    for note in table.attrs.get("notes") or []:
        st.caption(f"ℹ️ {note}")
    st.dataframe(table.head(500), hide_index=True, **WIDE)
    if len(table) > 500:
        st.caption(f"Showing the first 500 of {len(table):,} rows. The CSV contains all of them.")
    slug = re.sub(r"[^a-z0-9]+", "_", chart_type.lower()).strip("_")
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download chart PNG", result["png"], f"reviewmind_{slug}.png", "image/png",
        key="graph_builder_download_png", **WIDE)
    downloads[1].download_button(
        "Download chart data CSV", table.to_csv(index=False).encode("utf-8-sig"),
        f"reviewmind_{slug}_data.csv", "text/csv", key="graph_builder_download_csv", **WIDE)


def show_exports(data):
    results = st.session_state.get("analysis_results", {})
    st.subheader("Export center")
    frame = results.get("results_dataframe")
    if isinstance(frame, pd.DataFrame):
        st.download_button("Download analysis CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                           "reviewmind_analysis.csv", "text/csv", **WIDE)
    else:
        st.info("Run an analysis to create a predictions CSV.")
    report = build_report_pdf(data, results, profile_dataset(data))
    st.download_button("Download PDF report", report, "reviewmind_report.pdf", "application/pdf", **WIDE)


def show_terms():
    st.subheader("Key terms & guide")
    st.caption("Plain-language explanations of the methods used in ReviewMind AI.")
    for term, (definition, example) in TERMS.items():
        with st.expander(term):
            st.write(definition)
            st.caption(example)


if st.session_state.pop("demo_requested", False):
    st.session_state["source_choice"] = "Use demo dataset"

st.sidebar.markdown("## 🧠 ReviewMind AI")
st.sidebar.caption("Local customer feedback intelligence")
source = st.sidebar.radio("Start with", ["Upload a dataset", "Use demo dataset"], key="source_choice")
if source == "Use demo dataset":
    st.session_state["use_demo"] = True
else:
    st.session_state["use_demo"] = False
    uploaded = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        st.session_state["uploaded_file"] = uploaded
    else:
        # No file in the uploader: drop the previous file and the data parsed from it.
        for stale_key in ("uploaded_file", "original_data", "loaded_data_cache", "workbook_info"):
            st.session_state.pop(stale_key, None)

page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Analyze", "Visualizations", "Exports", "Key Terms"],
    key="page",
)
keep_widget_state()
try:
    data = load_current_data()
except (ValueError, FileNotFoundError, OSError) as error:
    data = None
    # Never leave the previous dataset behind for the chat or charts to read.
    for stale_key in ("original_data", "loaded_data_cache"):
        st.session_state.pop(stale_key, None)
    st.error(f"We could not load this dataset: {error}")

if data is None:
    show_empty_state()
else:
    if page == "Overview":
        show_overview(data)
        st.subheader("Next best step")
        st.write("Open **Analyze** to select a feedback column and run the preserved local ML pipeline.")
    elif page == "Analyze":
        show_analysis(data)
    elif page == "Visualizations":
        show_visualizations(data)
    elif page == "Exports":
        show_exports(data)
    else:
        show_terms()

render_chat_widget()

st.sidebar.divider()
st.sidebar.caption("No paid APIs • No arbitrary code execution • Original data preserved")