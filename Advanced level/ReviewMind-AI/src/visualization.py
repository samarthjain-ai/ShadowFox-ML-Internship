"""Dataset-aware chart builders used by the Streamlit visualization center."""

from __future__ import annotations

import re
import warnings
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_profiler import detect_date_columns, detect_rating_columns, parse_dates


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTE = ["#6d4fe6", "#14b8a6", "#f06a8b", "#f0a34e", "#3b82f6",
           "#84cc16", "#a855f7", "#ef4444", "#0ea5e9", "#eab308"]

CLASSIFY_SAMPLE_ROWS = 5000      # rows inspected when detecting column types
MAX_CATEGORIES = 50              # categories drawn on bar / pie charts
MAX_BOX_GROUPS = 30              # groups drawn on box / violin plots
MAX_COLOR_GROUPS = 8             # colour groups before the rest become "Other"
MAX_SCATTER_POINTS = 20000       # points drawn on scatter / bubble charts
MAX_VIOLIN_ROWS = 50000          # rows used to draw violin shapes
MAX_PAIR_ROWS = 2000             # rows used for pair plots
MAX_PAIR_COLUMNS = 6             # columns allowed in a pair plot
MAX_HEATMAP_COLUMNS = 25         # columns allowed in a correlation heatmap
LOW_CARDINALITY = 20             # numeric columns with this few values can act as categories

AGGREGATIONS = ["Count", "Sum", "Average", "Median", "Minimum", "Maximum"]
ADDITIVE_AGGREGATIONS = ["Count", "Sum"]
_AGG_FUNCTIONS = {"Sum": "sum", "Average": "mean", "Median": "median",
                  "Minimum": "min", "Maximum": "max"}

SORT_ORDERS = ["Value: high to low", "Value: low to high", "Label: A–Z", "Label: Z–A"]
DATE_GROUPINGS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def _spec(family, help_text, x=None, y=None, color=None, size=False, agg=None,
          sort=False, top_n=False, bins=False, multi=False, time=False,
          x_text="X-axis column", y_text="Value column (Y)",
          color_text="Color / group (optional)"):
    return {"family": family, "help": help_text, "x": x, "y": y, "color": color,
            "size": size, "agg": agg, "sort": sort, "top_n": top_n, "bins": bins,
            "multi": multi, "time": time, "x_text": x_text, "y_text": y_text,
            "color_text": color_text}


# x: category | category_optional | numeric | time        y: numeric | numeric_optional
# color: optional | required                              agg: all | additive
CHART_SPECS = {
    "Bar chart": _spec(
        "Categorical",
        "Compares a value across categories. Pick a numeric column and an aggregation, "
        "or leave the value empty to count rows.",
        x="category", y="numeric_optional", agg="all", sort=True, top_n=True,
        x_text="Category column (X)", y_text="Value column (Y, optional)"),
    "Horizontal bar chart": _spec(
        "Categorical",
        "Same as a bar chart with horizontal bars, which suits long category names.",
        x="category", y="numeric_optional", agg="all", sort=True, top_n=True,
        x_text="Category column", y_text="Value column (optional)"),
    "Pie chart": _spec(
        "Categorical",
        "Shows each category's share of the total. Uses Count or Sum so the slices are "
        "meaningful proportions.",
        x="category", y="numeric_optional", agg="additive", sort=True, top_n=True,
        x_text="Category column", y_text="Value column (optional)"),
    "Donut chart": _spec(
        "Categorical",
        "A pie chart with a hollow centre showing the total. Uses Count or Sum.",
        x="category", y="numeric_optional", agg="additive", sort=True, top_n=True,
        x_text="Category column", y_text="Value column (optional)"),
    "Stacked bar chart": _spec(
        "Categorical",
        "Splits each bar into parts by a second category. Uses Count or Sum so the "
        "stacked parts add up.",
        x="category", y="numeric_optional", color="required", agg="additive",
        sort=True, top_n=True, x_text="Category column (X)",
        y_text="Value column (optional)", color_text="Stack by (category)"),
    "Histogram": _spec(
        "Numerical",
        "Shows how the values of one numeric column are distributed.",
        x="numeric", color="optional", bins=True, x_text="Numeric column"),
    "Box plot": _spec(
        "Numerical",
        "Shows median, quartiles and outliers of a numeric column, optionally per group.",
        x="category_optional", y="numeric", sort=True, top_n=True,
        x_text="Group by (optional)", y_text="Numeric column"),
    "Violin plot": _spec(
        "Numerical",
        "Like a box plot, but also shows the shape of the distribution.",
        x="category_optional", y="numeric", sort=True, top_n=True,
        x_text="Group by (optional)", y_text="Numeric column"),
    "Scatter plot": _spec(
        "Numerical",
        "Shows the relationship between two numeric columns.",
        x="numeric", y="numeric", color="optional",
        x_text="X-axis (numeric)", y_text="Y-axis (numeric)"),
    "Bubble chart": _spec(
        "Numerical",
        "A scatter plot where a third numeric column controls the bubble size.",
        x="numeric", y="numeric", color="optional", size=True,
        x_text="X-axis (numeric)", y_text="Y-axis (numeric)"),
    "Line chart": _spec(
        "Time-based",
        "Shows how a value changes over dates or ordered numbers. Leave the value "
        "empty to count rows.",
        x="time", y="numeric_optional", color="optional", agg="all", time=True,
        x_text="X-axis (date or number)", y_text="Value column (Y, optional)"),
    "Area chart": _spec(
        "Time-based",
        "A line chart with the area under the line filled in.",
        x="time", y="numeric_optional", color="optional", agg="all", time=True,
        x_text="X-axis (date or number)", y_text="Value column (Y, optional)"),
    "Correlation heatmap": _spec(
        "Statistical",
        "Shows how strongly numeric columns move together (Pearson correlation). "
        "Correlation is not causation.",
        multi=True, x_text="Numeric columns"),
    "Pair plot": _spec(
        "Statistical",
        f"Scatter plots for every pair of up to {MAX_PAIR_COLUMNS} numeric columns. "
        "Large datasets are sampled.",
        multi=True, color="optional", x_text="Numeric columns"),
}
CHART_TYPES = list(CHART_SPECS)


def get_chart_spec(chart_type: str) -> dict:
    """Return the control requirements of one custom chart type."""

    if chart_type not in CHART_SPECS:
        raise ValueError(
            f"'{chart_type}' is not a supported chart type. "
            f"Choose one of: {', '.join(CHART_TYPES)}."
        )
    return CHART_SPECS[chart_type]


def allowed_aggregations(chart_type: str, has_value: bool = True) -> list[str]:
    """Aggregations that are valid for a chart type and the chosen value column."""

    spec = get_chart_spec(chart_type)
    if spec["agg"] is None or not has_value:
        return ["Count"]
    return list(AGGREGATIONS) if spec["agg"] == "all" else list(ADDITIVE_AGGREGATIONS)


# ---------------------------------------------------------------------------
# Export + legacy helpers (unchanged behaviour)
# ---------------------------------------------------------------------------

def chart_bytes(figure) -> bytes:
    """Serialize a matplotlib figure for a Streamlit download button."""

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def default_date_grouping(data: pd.DataFrame, date_column: str) -> str:
    """Choose a readable default date grain for a graph builder."""

    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' was not found.")
    dates = parse_dates(data[date_column])
    unique_dates = int(dates.dropna().dt.normalize().nunique())
    return "month" if unique_dates > 14 else "day"


def aggregate_date_values(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    date_grouping: str = "month",
) -> pd.DataFrame:
    """Aggregate valid numeric values by a parsed date column."""

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The current dataset is empty.")
    for column, label in ((date_column, "Date"), (value_column, "Rating")):
        if column not in data.columns:
            raise ValueError(
                f"The required {label} column is unavailable. "
                f"Select a valid {label.lower()} column."
            )
    if date_grouping not in {"day", "week", "month"}:
        raise ValueError("Date grouping must be Day, Week, or Month.")

    dates = parse_dates(data[date_column])
    values = pd.to_numeric(data[value_column], errors="coerce")
    valid = dates.notna() & values.notna() & values.between(1, 5)
    if not valid.any():
        raise ValueError(
            "No rows contain both a valid date and a numeric 1–5 rating. "
            "Check the selected columns, missing values, and rating range."
        )
    excluded = {
        "total_rows": int(len(data)),
        "valid_rows": int(valid.sum()),
        "invalid_or_missing_date": int(dates.isna().sum()),
        "invalid_or_missing_rating": int((values.isna() | ~values.between(1, 5)).sum()),
    }

    valid_dates = dates.loc[valid]
    if date_grouping == "day":
        groups = valid_dates.dt.strftime("%Y-%m-%d")
    elif date_grouping == "week":
        groups = valid_dates.dt.to_period("W").map(
            lambda period: period.start_time.strftime("%Y-%m-%d")
        )
    else:
        groups = valid_dates.dt.to_period("M").astype(str)

    working = pd.DataFrame({
        "date_group": groups.to_numpy(),
        "rating": values.loc[valid].to_numpy(),
    })
    aggregated = (
        working.groupby("date_group", as_index=False)
        .agg(average_rating=("rating", "mean"), record_count=("rating", "size"))
        .sort_values("date_group")
        .reset_index(drop=True)
    )
    aggregated["average_rating"] = aggregated["average_rating"].round(3)
    aggregated.attrs["row_summary"] = excluded
    return aggregated


# ---------------------------------------------------------------------------
# Automatic column detection
# ---------------------------------------------------------------------------

_ID_TOKEN = re.compile(
    r"(^|[\s_\-./])(id|uuid|guid|key|index|idx|no|num|number|code|ref|sku)($|[\s_\-./])",
    re.IGNORECASE,
)
_ID_SUFFIX = re.compile(r"(?<=[a-z])(Id|ID)$")


def _name_looks_like_identifier(name) -> bool:
    text = str(name)
    return bool(_ID_TOKEN.search(text) or _ID_SUFFIX.search(text)
                or text.lower().startswith("unnamed"))


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse dates safely, preferring the project's own ``parse_dates`` helper."""

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        parsed = None
        try:
            candidate = parse_dates(series)
            if (isinstance(candidate, pd.Series) and len(candidate) == len(series)
                    and pd.api.types.is_datetime64_any_dtype(candidate)):
                parsed = candidate
        except Exception:  # fall back to the plain pandas parser
            parsed = None
        if parsed is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _numeric_kind(numbers: pd.Series, name) -> tuple[str, bool]:
    """Return ``("numeric" | "identifier", is_low_cardinality)`` for numeric values."""

    count = len(numbers)
    unique = int(numbers.nunique())
    ratio = unique / count if count else 0.0
    integer_like = bool((numbers % 1 == 0).all())
    if integer_like and _name_looks_like_identifier(name) and ratio >= 0.5:
        return "identifier", False
    if integer_like and count >= 20 and unique == count:
        ordered = np.sort(numbers.to_numpy(dtype=float))
        if bool((np.diff(ordered) == 1).all()):          # 1, 2, 3 ... row numbers
            return "identifier", False
        if ratio >= 0.95 and float(ordered[0]) >= 1e8:   # phone-number-like values
            return "identifier", False
    return "numeric", unique <= LOW_CARDINALITY


def _classify_series(series: pd.Series, name) -> tuple[str, bool]:
    """Classify one column from (a sample of) its actual values."""

    if pd.api.types.is_bool_dtype(series):
        return ("boolean", False) if series.notna().any() else ("empty", False)
    if pd.api.types.is_datetime64_any_dtype(series):
        return ("datetime", False) if series.notna().any() else ("empty", False)
    non_null = series.dropna()
    if non_null.empty:
        return "empty", False
    if pd.api.types.is_numeric_dtype(series):
        numbers = pd.to_numeric(non_null, errors="coerce").astype("float64")
        numbers = numbers.replace([np.inf, -np.inf], np.nan).dropna()
        if numbers.empty:
            return "empty", False
        return _numeric_kind(numbers, name)

    strings = non_null.astype(str).str.strip()
    strings = strings[strings != ""]
    if strings.empty:
        return "empty", False
    if set(strings.str.lower().unique()) <= {"true", "false", "yes", "no"}:
        return "boolean", False

    leading_zero = bool(strings.str.match(r"^0\d+$").any())
    numbers = pd.to_numeric(strings, errors="coerce")
    if not leading_zero and numbers.notna().mean() >= 0.95:
        return _numeric_kind(numbers.replace([np.inf, -np.inf], np.nan).dropna(), name)

    if strings.str.contains(r"\d").mean() >= 0.8:
        parsed = _to_datetime(strings.reset_index(drop=True))
        if parsed.notna().mean() >= 0.8:
            return "datetime", False

    unique = int(strings.nunique())
    ratio = unique / len(strings)
    average_length = float(strings.str.len().mean())
    average_words = float(strings.str.split().str.len().mean())
    if average_length >= 50 or average_words >= 7:
        return "text", False
    if _name_looks_like_identifier(name) and ratio >= 0.5:
        return "identifier", False
    if ratio >= 0.95 and len(strings) >= 20 and unique > 50:
        return "identifier", False
    return "categorical", False


def classify_columns(data: pd.DataFrame) -> dict:
    """Group the columns of ``data`` by what their values actually contain.

    Only a bounded random sample is inspected and the DataFrame is never
    modified. Keys: numeric, categorical, datetime, text, boolean, identifier,
    empty, low_cardinality_numeric, and ``order`` (all columns in dataset order).
    """

    result = {key: [] for key in ("numeric", "categorical", "datetime", "text", "boolean",
                                  "identifier", "empty", "low_cardinality_numeric", "order")}
    if not isinstance(data, pd.DataFrame) or data.shape[1] == 0:
        return result
    sample = data
    if len(data) > CLASSIFY_SAMPLE_ROWS:
        sample = data.sample(n=CLASSIFY_SAMPLE_ROWS, random_state=0)
    seen = set()
    for position, name in enumerate(data.columns):
        if name in seen:
            continue
        seen.add(name)
        result["order"].append(name)
        try:
            kind, low = _classify_series(sample.iloc[:, position], name)
        except Exception:  # an odd column must never break the whole builder
            kind, low = "empty", False
        result[kind].append(name)
        if kind == "numeric" and low:
            result["low_cardinality_numeric"].append(name)
    return result


def _ordered(classes: dict, names) -> list:
    wanted = set(names)
    return [name for name in classes.get("order", []) if name in wanted]


def column_options(classes: dict, chart_type: str) -> dict:
    """Compatible columns for each control of ``chart_type``."""

    spec = get_chart_spec(chart_type)
    numeric = _ordered(classes, classes.get("numeric", []))
    category = _ordered(classes, list(classes.get("categorical", []))
                        + list(classes.get("boolean", []))
                        + list(classes.get("low_cardinality_numeric", [])))
    time = _ordered(classes, list(classes.get("datetime", [])) + numeric)
    x_pool = {"category": category, "category_optional": category,
              "numeric": numeric, "time": time}.get(spec["x"], [])
    return {
        "x": x_pool,
        "y": numeric if spec["y"] else [],
        "color": category if spec["color"] else [],
        "size": numeric if spec["size"] else [],
        "multi": numeric if spec["multi"] else [],
    }


def available_chart_types(classes: dict) -> tuple[list[str], dict]:
    """Return ``(compatible chart types, {incompatible type: reason})``."""

    counts = {
        "category": len(column_options(classes, "Bar chart")["x"]),
        "numeric": len(classes.get("numeric", [])),
        "time": len(column_options(classes, "Line chart")["x"]),
    }
    needs = {
        "Bar chart": ("category", 1, "needs a categorical column"),
        "Horizontal bar chart": ("category", 1, "needs a categorical column"),
        "Pie chart": ("category", 1, "needs a categorical column"),
        "Donut chart": ("category", 1, "needs a categorical column"),
        "Stacked bar chart": ("category", 2, "needs two categorical columns"),
        "Histogram": ("numeric", 1, "needs a numeric column"),
        "Box plot": ("numeric", 1, "needs a numeric column"),
        "Violin plot": ("numeric", 1, "needs a numeric column"),
        "Scatter plot": ("numeric", 2, "needs two numeric columns"),
        "Bubble chart": ("numeric", 3, "needs three numeric columns"),
        "Line chart": ("time", 1, "needs a date or numeric column"),
        "Area chart": ("time", 1, "needs a date or numeric column"),
        "Correlation heatmap": ("numeric", 2, "needs two numeric columns"),
        "Pair plot": ("numeric", 2, "needs two numeric columns"),
    }
    available, unavailable = [], {}
    for chart_type in CHART_TYPES:
        kind, minimum, reason = needs[chart_type]
        if counts[kind] >= minimum:
            available.append(chart_type)
        else:
            unavailable[chart_type] = reason
    return available, unavailable


def column_type_table(data: pd.DataFrame, classes: dict) -> pd.DataFrame:
    """A small table describing the detected type of every column."""

    labels = {"numeric": "Numeric", "categorical": "Categorical", "datetime": "Date / time",
              "text": "Text", "boolean": "Boolean", "identifier": "Identifier",
              "empty": "Empty"}
    kind_of = {}
    for kind, label in labels.items():
        for name in classes.get(kind, []):
            kind_of[name] = label
    rows = []
    for name in classes.get("order", []):
        label = kind_of.get(name, "Other")
        if name in classes.get("low_cardinality_numeric", []):
            label += " (few values — also usable as a category)"
        try:
            missing = int(data[name].isna().sum()) if not isinstance(data[name], pd.DataFrame) else 0
        except Exception:
            missing = 0
        rows.append({"Column": str(name), "Detected type": label, "Missing values": missing})
    return pd.DataFrame(rows, columns=["Column", "Detected type", "Missing values"])


# ---------------------------------------------------------------------------
# Safe value access
# ---------------------------------------------------------------------------

def _clean_optional(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _column(data: pd.DataFrame, name) -> pd.Series:
    if name is None or name not in data.columns:
        raise ValueError(f"Column '{name}' was not found in the current dataset.")
    series = data[name]
    return series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series


def _numeric(data: pd.DataFrame, name) -> pd.Series:
    """Numeric view of a column; unusable values become NaN (no copy of ``data``)."""

    series = _column(data, name)
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"Column '{name}' is True/False, not numeric. Choose a numeric column.")
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(object)
    numbers = pd.to_numeric(series, errors="coerce").astype("float64")
    numbers = numbers.replace([np.inf, -np.inf], np.nan)
    if numbers.notna().sum() == 0:
        raise ValueError(f"Column '{name}' has no numeric values, so it cannot be used here.")
    return numbers


def _labels(data: pd.DataFrame, name) -> tuple[pd.Series, pd.Series]:
    """Text labels for a categorical column plus a mask of usable rows."""

    series = _column(data, name)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numbers = pd.to_numeric(series, errors="coerce").astype("float64")
        numbers = numbers.replace([np.inf, -np.inf], np.nan)
        valid = numbers.notna()
        if valid.any() and bool((numbers[valid] % 1 == 0).all()):
            labels = numbers.round().astype("Int64").astype(str)
        else:
            labels = numbers.astype(str)
        return labels, valid
    labels = series.astype(str).str.strip()
    valid = series.notna() & (labels != "")
    return labels, valid


def _top_groups(labels: pd.Series, limit: int = MAX_COLOR_GROUPS):
    """Keep the ``limit`` most common labels and fold the rest into 'Other'."""

    counts = labels.value_counts()
    if len(counts) <= limit:
        return labels, [str(item) for item in counts.index], None
    keep = [str(item) for item in counts.index[:limit]]
    grouped = labels.where(labels.isin(keep), "Other")
    return grouped, keep + ["Other"], int(len(counts))


def _colors(count: int) -> list[str]:
    if count <= len(PALETTE):
        return PALETTE[:count]
    return sns.color_palette("husl", count).as_hex()


def _shorten(text, limit: int = 32) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(value)
    if abs(number - round(number)) < 1e-9 or abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:,.2f}"


def _measure_name(aggregation: str, value_column) -> str:
    if value_column is None:
        return "Number of records"
    if aggregation == "Count":
        return f"Count of {value_column}"
    return f"{aggregation} of {value_column}"


def _aggregate(frame: pd.DataFrame, group_columns: list, value_column, aggregation: str):
    grouped = frame.groupby(group_columns, sort=False, observed=True)
    if aggregation == "Count":
        out = grouped.size() if value_column is None else grouped[value_column].count()
    else:
        out = grouped[value_column].agg(_AGG_FUNCTIONS[aggregation])
    return out.rename("value").reset_index()


def _sort_table(table: pd.DataFrame, label_column: str, value_column: str, sort_order: str):
    if sort_order not in SORT_ORDERS:
        raise ValueError(f"Sort order must be one of: {', '.join(SORT_ORDERS)}.")
    if sort_order == "Value: high to low":
        return table.sort_values(value_column, ascending=False, kind="mergesort")
    if sort_order == "Value: low to high":
        return table.sort_values(value_column, ascending=True, kind="mergesort")
    labels = table[label_column]
    numbers = pd.to_numeric(labels, errors="coerce")
    key = numbers.to_numpy() if numbers.notna().all() else labels.str.lower().to_numpy()
    positions = np.argsort(key, kind="mergesort")
    if sort_order == "Label: Z–A":
        positions = positions[::-1]
    return table.iloc[positions]


def _sample(frame: pd.DataFrame, limit: int, notes: list, what: str) -> pd.DataFrame:
    if len(frame) > limit:
        notes.append(
            f"Performance sampling: {what} uses a random sample of {limit:,} of "
            f"{len(frame):,} valid rows (fixed seed). The data is otherwise unchanged."
        )
        return frame.sample(n=limit, random_state=42).sort_index()
    return frame


def _finish(figure, axis, default_title, default_x, default_y, title, x_label, y_label):
    axis.set_title(title or default_title)
    if x_label or default_x:
        axis.set_xlabel(x_label or default_x)
    if y_label or default_y:
        axis.set_ylabel(y_label or default_y)
    try:
        figure.tight_layout()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Category-based charts: bar, horizontal bar, pie, donut
# ---------------------------------------------------------------------------

def _category_table(data, x, y, aggregation, sort_order, top_n, notes, lump_other=False):
    labels, valid = _labels(data, x)
    values = None
    if y is not None:
        values = _numeric(data, y)
        valid = valid & values.notna()
    used = int(valid.sum())
    if used == 0:
        raise ValueError(
            "No rows contain both a usable category and a numeric value. "
            "Check the selected columns and missing values."
        )
    frame = pd.DataFrame({"category": labels[valid].to_numpy()})
    if values is not None:
        frame["number"] = values[valid].to_numpy()
    table = _aggregate(frame, ["category"], "number" if y is not None else None, aggregation)

    categories = len(table)
    limit = int(top_n) if top_n else None
    if limit is None and categories > MAX_CATEGORIES:
        limit = MAX_CATEGORIES
        notes.append(
            f"'{x}' has {categories:,} categories, so the {limit} largest are shown. "
            "Use “Show only the top N” to choose a different number."
        )
        explicit = False
    else:
        explicit = True
    table = table.sort_values("value", ascending=False, kind="mergesort")
    if limit and categories > limit:
        kept, rest = table.head(limit), table.iloc[limit:]
        if lump_other and aggregation in ADDITIVE_AGGREGATIONS:
            kept = pd.concat([kept, pd.DataFrame(
                {"category": ["All other categories"], "value": [rest["value"].sum()]})],
                ignore_index=True)
            if explicit:
                notes.append(f"Showing the top {limit} of {categories:,} categories; "
                             f"the other {len(rest):,} are grouped together.")
        elif explicit:
            notes.append(f"Showing the top {limit} of {categories:,} categories by value.")
        table = kept
    table = _sort_table(table.reset_index(drop=True), "category", "value", sort_order)
    return table.reset_index(drop=True), {"total_rows": int(len(data)), "valid_rows": used}


def _draw_bars(table, measure, x_name, horizontal, title, x_label, y_label):
    count = len(table)
    labels = [_shorten(item) for item in table["category"]]
    values = table["value"].to_numpy(dtype=float)
    positions = np.arange(count)
    if horizontal:
        figure, axis = plt.subplots(figsize=(9, max(4.0, 0.38 * count + 1.8)))
        bars = axis.barh(positions, values, color=PALETTE[0])
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        defaults = (measure, x_name)
    else:
        figure, axis = plt.subplots(figsize=(max(7.5, min(18.0, 0.55 * count + 3.0)), 5.2))
        bars = axis.bar(positions, values, color=PALETTE[0])
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35 if count > 5 else 0,
                             ha="right" if count > 5 else "center")
        defaults = (x_name, measure)
    if count <= 30:
        axis.bar_label(bars, labels=[_fmt(v) for v in values], padding=3, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    _finish(figure, axis, f"{measure} by {x_name}", defaults[0], defaults[1],
            title, x_label, y_label)
    return figure


def _draw_pie(table, measure, x_name, donut, title):
    values = table["value"].to_numpy(dtype=float)
    total = float(values.sum())
    count = len(values)
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    wedge_props = {"linewidth": 1, "edgecolor": "white"}
    if donut:
        wedge_props["width"] = 0.45
    wedges, _, _ = axis.pie(
        values, labels=None, startangle=90, counterclock=False, colors=_colors(count),
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        pctdistance=0.78 if donut else 0.65, wedgeprops=wedge_props,
        textprops={"fontsize": 9, "color": "white" if donut else "black"},
    )
    axis.axis("equal")
    if donut:
        axis.text(0, 0, f"Total\n{_fmt(total)}", ha="center", va="center",
                  fontsize=12, fontweight="bold")
    legend = [f"{_shorten(label)} — {_fmt(value)} ({value / total * 100:.1f}%)"
              for label, value in zip(table["category"], values)]
    axis.legend(wedges, legend, title=x_name, loc="center left", bbox_to_anchor=(1, 0.5),
                frameon=False, fontsize=8 if count > 15 else 9)
    axis.set_title(title or f"{measure} by {x_name}")
    try:
        figure.tight_layout()
    except Exception:
        pass
    return figure


def _category_chart(data, chart_type, x, y, aggregation, sort_order, top_n,
                    title, x_label, y_label, notes):
    pie = chart_type in ("Pie chart", "Donut chart")
    table, summary = _category_table(data, x, y, aggregation, sort_order, top_n, notes,
                                     lump_other=pie)
    measure = _measure_name(aggregation, y)
    if pie:
        if (table["value"] < 0).any():
            raise ValueError("Pie and donut charts need non-negative values. "
                             "Choose a bar chart or a different value column.")
        if float(table["value"].sum()) <= 0:
            raise ValueError("The values add up to zero, so there are no proportions to show.")
        figure = _draw_pie(table, measure, x, chart_type == "Donut chart", title)
        table["percent_of_total"] = (table["value"] / table["value"].sum() * 100).round(2)
    else:
        figure = _draw_bars(table, measure, x, chart_type == "Horizontal bar chart",
                            title, x_label, y_label)
    table = table.rename(columns={"category": str(x), "value": measure})
    return figure, table, summary


def _stacked_bar(data, x, y, color, aggregation, sort_order, top_n, title, x_label,
                 y_label, notes):
    if x == color:
        raise ValueError("Choose two different columns for the bars and the stacked parts.")
    x_labels, x_valid = _labels(data, x)
    c_labels, c_valid = _labels(data, color)
    valid = x_valid & c_valid
    values = None
    if y is not None:
        values = _numeric(data, y)
        valid = valid & values.notna()
    used = int(valid.sum())
    if used == 0:
        raise ValueError("No rows contain usable values in all selected columns.")
    frame = pd.DataFrame({"x": x_labels[valid].to_numpy(), "part": c_labels[valid].to_numpy()})
    if values is not None:
        frame["number"] = values[valid].to_numpy()
    long = _aggregate(frame, ["x", "part"], "number" if y is not None else None, aggregation)
    wide = long.pivot(index="x", columns="part", values="value").fillna(0.0)

    # Parts: keep the largest groups, fold the rest into "Other" (valid for Count/Sum).
    part_totals = wide.sum(axis=0).sort_values(ascending=False)
    if len(part_totals) > MAX_COLOR_GROUPS:
        keep = list(part_totals.index[:MAX_COLOR_GROUPS])
        other = wide.drop(columns=keep).sum(axis=1)
        notes.append(f"'{color}' has {len(part_totals):,} groups; the {MAX_COLOR_GROUPS} "
                     "largest are shown and the rest are grouped as 'Other'.")
        wide = wide[keep].assign(Other=other)
    wide = wide[list(wide.sum(axis=0).sort_values(ascending=False).index)]

    totals = wide.sum(axis=1)
    categories = len(totals)
    limit = int(top_n) if top_n else (MAX_CATEGORIES if categories > MAX_CATEGORIES else None)
    if limit and categories > limit:
        wide = wide.loc[totals.sort_values(ascending=False, kind="mergesort").index[:limit]]
        notes.append(f"Showing the top {limit} of {categories:,} '{x}' categories by total.")
    order_frame = pd.DataFrame({"category": wide.index, "value": wide.sum(axis=1).to_numpy()})
    order = _sort_table(order_frame, "category", "value", sort_order)["category"].tolist()
    wide = wide.loc[order]

    measure = _measure_name(aggregation, y)
    figure, axis = plt.subplots(
        figsize=(max(8.0, min(18.0, 0.6 * len(wide) + 4.0)), 5.4))
    positions = np.arange(len(wide))
    bottoms = np.zeros(len(wide))
    for column, colour in zip(wide.columns, _colors(len(wide.columns))):
        heights = wide[column].to_numpy(dtype=float)
        axis.bar(positions, heights, bottom=bottoms, color=colour, label=_shorten(column))
        bottoms += heights
    axis.set_xticks(positions)
    axis.set_xticklabels([_shorten(item) for item in wide.index],
                         rotation=35 if len(wide) > 5 else 0,
                         ha="right" if len(wide) > 5 else "center")
    axis.legend(title=str(color), loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _finish(figure, axis, f"{measure} by {x}, stacked by {color}", str(x), measure,
            title, x_label, y_label)
    table = wide.reset_index().rename(columns={"x": str(x)})
    table.columns.name = None
    return figure, table, {"total_rows": int(len(data)), "valid_rows": used}


# ---------------------------------------------------------------------------
# Numerical charts: histogram, box, violin, scatter, bubble
# ---------------------------------------------------------------------------

def _histogram(data, x, color, bins, title, x_label, y_label, notes):
    values = _numeric(data, x)
    valid = values.notna()
    if color:
        labels, color_valid = _labels(data, color)
        valid = valid & color_valid
    used = int(valid.sum())
    if used == 0:
        raise ValueError("No rows contain usable values in the selected columns.")
    frame = pd.DataFrame({"value": values[valid].to_numpy()})
    order = []
    if color:
        grouped, order, total = _top_groups(pd.Series(labels[valid].to_numpy()))
        frame["group"] = grouped.to_numpy()
        if total:
            notes.append(f"'{color}' has {total:,} groups; the {MAX_COLOR_GROUPS} most common "
                         "are shown and the rest are grouped as 'Other'.")
    edges = np.histogram_bin_edges(frame["value"], bins=bins)
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    table = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:]})
    if color:
        arrays = [frame.loc[frame["group"] == name, "value"].to_numpy() for name in order]
        axis.hist(arrays, bins=edges, stacked=True, color=_colors(len(order)),
                  label=[_shorten(name) for name in order], edgecolor="white", linewidth=0.5)
        axis.legend(title=str(color), loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        for name, array in zip(order, arrays):
            table[str(name)] = np.histogram(array, bins=edges)[0]
    else:
        axis.hist(frame["value"], bins=edges, color=PALETTE[0], edgecolor="white", linewidth=0.5)
        table["count"] = np.histogram(frame["value"], bins=edges)[0]
    axis.spines[["top", "right"]].set_visible(False)
    _finish(figure, axis, f"Distribution of {x}", str(x), "Number of records",
            title, x_label, y_label)
    return figure, table, {"total_rows": int(len(data)), "valid_rows": used}


def _box_or_violin(data, chart_type, x, y, sort_order, top_n, title, x_label, y_label, notes):
    values = _numeric(data, y)
    valid = values.notna()
    if x is not None:
        labels, label_valid = _labels(data, x)
        valid = valid & label_valid
    used = int(valid.sum())
    if used == 0:
        raise ValueError("No rows contain usable values in the selected columns.")
    frame = pd.DataFrame({"value": values[valid].to_numpy()})
    frame["group"] = labels[valid].to_numpy() if x is not None else str(y)

    grouped = frame.groupby("group")["value"]
    stats = pd.DataFrame({
        "count": grouped.count(), "mean": grouped.mean(), "std": grouped.std(),
        "min": grouped.min(), "q1": grouped.quantile(0.25), "median": grouped.median(),
        "q3": grouped.quantile(0.75), "max": grouped.max(),
    }).reset_index()
    groups = len(stats)
    limit = int(top_n) if top_n else (MAX_BOX_GROUPS if groups > MAX_BOX_GROUPS else None)
    if limit and groups > limit:
        stats = stats.sort_values("count", ascending=False, kind="mergesort").head(limit)
        notes.append(f"'{x}' has {groups:,} groups; the {limit} with the most records are shown.")
    stats = _sort_table(stats.reset_index(drop=True), "group", "median", sort_order)
    stats = stats.reset_index(drop=True)
    order = stats["group"].tolist()
    frame = frame[frame["group"].isin(order)]

    figure, axis = plt.subplots(figsize=(max(7.0, min(18.0, 0.7 * len(order) + 4.0)), 5.2))
    if chart_type == "Box plot":
        arrays = [frame.loc[frame["group"] == name, "value"].to_numpy() for name in order]
        parts = axis.boxplot(arrays, positions=np.arange(1, len(order) + 1), patch_artist=True,
                             widths=0.6, medianprops={"color": "#111827", "linewidth": 1.6},
                             flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
        for patch, colour in zip(parts["boxes"], _colors(len(order))):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        axis.set_xticks(np.arange(1, len(order) + 1))
        axis.set_xticklabels([_shorten(name) for name in order],
                             rotation=35 if len(order) > 5 else 0,
                             ha="right" if len(order) > 5 else "center")
    else:
        drawn = _sample(frame, MAX_VIOLIN_ROWS, notes, "the violin shapes")
        sns.violinplot(data=drawn, x="group", y="value", order=order, color=PALETTE[0],
                       inner="quartile", cut=0, ax=axis)
        axis.set_xticks(np.arange(len(order)))
        axis.set_xticklabels([_shorten(name) for name in order],
                             rotation=35 if len(order) > 5 else 0,
                             ha="right" if len(order) > 5 else "center")
    axis.spines[["top", "right"]].set_visible(False)
    kind = "Box plot" if chart_type == "Box plot" else "Violin plot"
    default_title = f"{kind} of {y}" + (f" by {x}" if x is not None else "")
    _finish(figure, axis, default_title, str(x) if x is not None else "", str(y),
            title, x_label, y_label)
    stats = stats.rename(columns={"group": str(x) if x is not None else "column"})
    return figure, stats, {"total_rows": int(len(data)), "valid_rows": used}


def _scatter_or_bubble(data, chart_type, x, y, color, size, title, x_label, y_label, notes):
    if x == y:
        raise ValueError("Choose two different numeric columns for X and Y.")
    xs, ys = _numeric(data, x), _numeric(data, y)
    valid = xs.notna() & ys.notna()
    sizes = None
    if chart_type == "Bubble chart":
        sizes = _numeric(data, size)
        valid = valid & sizes.notna()
    if color:
        labels, color_valid = _labels(data, color)
        valid = valid & color_valid
    used = int(valid.sum())
    if used == 0:
        raise ValueError("No rows contain usable values in all selected columns.")
    frame = pd.DataFrame({str(x): xs[valid].to_numpy(), str(y): ys[valid].to_numpy()})
    if sizes is not None:
        frame[str(size)] = sizes[valid].to_numpy()
    if color:
        grouped, order, total = _top_groups(pd.Series(labels[valid].to_numpy()))
        frame[str(color)] = grouped.to_numpy()
        if total:
            notes.append(f"'{color}' has {total:,} groups; the {MAX_COLOR_GROUPS} most common "
                         "are colored and the rest are grouped as 'Other'.")
    frame = _sample(frame, MAX_SCATTER_POINTS, notes, "this chart")

    if sizes is not None:
        raw = frame[str(size)].to_numpy(dtype=float)
        low, high = raw.min(), raw.max()
        area = 40 + 760 * (raw - low) / (high - low) if high > low else np.full(len(raw), 160.0)
        notes.append(f"Bubble area is scaled between the smallest and largest '{size}' value "
                     f"({_fmt(low)} to {_fmt(high)}).")
    else:
        area = np.full(len(frame), 22.0)
    alpha = 0.55 if len(frame) > 200 else 0.75
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    if color:
        groups = frame[str(color)].to_numpy()
        names = [name for name in order if name in set(groups)]
        for name, colour in zip(names, _colors(len(names))):
            mask = groups == name
            axis.scatter(frame.loc[mask, str(x)], frame.loc[mask, str(y)], s=area[mask],
                         color=colour, alpha=alpha, edgecolors="white", linewidths=0.4,
                         label=_shorten(name))
        axis.legend(title=str(color), loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    else:
        axis.scatter(frame[str(x)], frame[str(y)], s=area, color=PALETTE[0], alpha=alpha,
                     edgecolors="white", linewidths=0.4)
    axis.spines[["top", "right"]].set_visible(False)
    kind = "Bubble chart" if sizes is not None else "Scatter plot"
    _finish(figure, axis, f"{y} vs {x}" + (f" (bubble size: {size})" if sizes is not None else ""),
            str(x), str(y), title, x_label, y_label)
    return figure, frame.reset_index(drop=True), {
        "total_rows": int(len(data)), "valid_rows": used}


# ---------------------------------------------------------------------------
# Time-based charts: line and area
# ---------------------------------------------------------------------------

def _time_axis(data, x):
    series = _column(data, x)
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"Column '{x}' is True/False and cannot be used as a line or area X-axis.")
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date", _to_datetime(series)
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", _numeric(data, x)
    present = int(series.notna().sum())
    numbers = pd.to_numeric(series, errors="coerce")
    if present and numbers.notna().sum() / present >= 0.9:
        return "numeric", numbers.astype("float64").replace([np.inf, -np.inf], np.nan)
    dates = _to_datetime(series)
    if present and dates.notna().sum() / present >= 0.5:
        return "date", dates
    raise ValueError(f"Column '{x}' has neither dates nor ordered numbers, so it cannot be "
                     "used on the X-axis of a line or area chart.")


def _time_chart(data, chart_type, x, y, color, aggregation, date_grouping,
                title, x_label, y_label, notes):
    kind, axis_values = _time_axis(data, x)
    valid = axis_values.notna()
    values = None
    if y is not None:
        values = _numeric(data, y)
        valid = valid & values.notna()
    if color:
        labels, color_valid = _labels(data, color)
        valid = valid & color_valid
    used = int(valid.sum())
    if used == 0:
        raise ValueError("No rows contain usable values in all selected columns.")

    grouping = str(date_grouping or "month").lower()
    if kind == "date":
        if grouping not in DATE_GROUPINGS:
            raise ValueError(f"Date grouping must be one of: {', '.join(g.title() for g in DATE_GROUPINGS)}.")
        axis_column = f"{x} ({grouping})"
        x_data = axis_values[valid].dt.to_period(DATE_GROUPINGS[grouping]).dt.start_time
    else:
        axis_column = str(x)
        x_data = axis_values[valid]
    frame = pd.DataFrame({"x": x_data.to_numpy()})
    if values is not None:
        frame["number"] = values[valid].to_numpy()
    group_columns = ["x"]
    if color:
        frame["group"] = labels[valid].to_numpy()
        counts = frame["group"].value_counts()
        if len(counts) > MAX_COLOR_GROUPS:
            keep = set(counts.index[:MAX_COLOR_GROUPS])
            dropped = int((~frame["group"].isin(keep)).sum())
            frame = frame[frame["group"].isin(keep)]
            notes.append(f"'{color}' has {len(counts):,} groups; the {MAX_COLOR_GROUPS} with the "
                         f"most records are drawn ({dropped:,} rows in other groups are not shown).")
        group_columns.append("group")
    table = _aggregate(frame, group_columns, "number" if y is not None else None, aggregation)
    table = table.sort_values(group_columns, kind="mergesort").reset_index(drop=True)
    measure = _measure_name(aggregation, y)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    area = chart_type == "Area chart"
    if color:
        names = list(table.groupby("group")["value"].count().sort_values(ascending=False).index)
        palette = dict(zip(names, _colors(len(names))))
        if area and aggregation in ADDITIVE_AGGREGATIONS:
            wide = table.pivot(index="x", columns="group", values="value").fillna(0.0)[names]
            axis.stackplot(wide.index, [wide[name].to_numpy() for name in names],
                           labels=[_shorten(name) for name in names],
                           colors=[palette[name] for name in names], alpha=0.85)
        else:
            for name in names:
                part = table[table["group"] == name]
                axis.plot(part["x"], part["value"], color=palette[name], linewidth=2,
                          marker="o" if len(part) <= 40 else None, label=_shorten(name))
                if area:
                    axis.fill_between(part["x"], part["value"], color=palette[name], alpha=0.2)
        axis.legend(title=str(color), loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    else:
        axis.plot(table["x"], table["value"], color=PALETTE[0], linewidth=2,
                  marker="o" if len(table) <= 40 else None)
        if area:
            axis.fill_between(table["x"], table["value"], color=PALETTE[0], alpha=0.3)
    if kind == "date":
        axis.tick_params(axis="x", rotation=30)
    axis.spines[["top", "right"]].set_visible(False)
    suffix = f" ({grouping})" if kind == "date" else ""
    _finish(figure, axis, f"{measure} by {x}{suffix}", axis_column, measure,
            title, x_label, y_label)

    out = table.rename(columns={"x": axis_column, "group": str(color), "value": measure})
    if kind == "date":
        formats = {"day": "%Y-%m-%d", "week": "%Y-%m-%d", "month": "%Y-%m",
                   "quarter": "%Y-%m", "year": "%Y"}
        out[axis_column] = pd.to_datetime(out[axis_column]).dt.strftime(formats[grouping])
    return figure, out, {"total_rows": int(len(data)), "valid_rows": used}


# ---------------------------------------------------------------------------
# Statistical charts: correlation heatmap and pair plot
# ---------------------------------------------------------------------------

def _usable_numeric_frame(data, columns, notes):
    """Numeric working frame; skips unusable and constant columns with explanations."""

    frame = {}
    for name in columns:
        try:
            numbers = _numeric(data, name)
        except ValueError:
            notes.append(f"Skipped '{name}': it has no numeric values.")
            continue
        if numbers.nunique(dropna=True) <= 1:
            notes.append(f"Skipped '{name}': every value is the same, so correlation is undefined.")
            continue
        frame[str(name)] = numbers
    return pd.DataFrame(frame)


def _heatmap(data, columns, title, notes):
    if not columns:
        columns = classify_columns(data)["numeric"]
    columns = list(dict.fromkeys(columns))
    if len(columns) > MAX_HEATMAP_COLUMNS:
        notes.append(f"Using the first {MAX_HEATMAP_COLUMNS} of {len(columns)} selected columns.")
        columns = columns[:MAX_HEATMAP_COLUMNS]
    frame = _usable_numeric_frame(data, columns, notes)
    if frame.shape[1] < 2:
        raise ValueError("A correlation heatmap needs at least two numeric columns that contain "
                         "more than one distinct value.")
    if int(frame.dropna().shape[0]) < 2 and int(frame.notna().sum().min()) < 2:
        raise ValueError("There are not enough rows with numeric values to compute correlations.")
    correlation = frame.corr()
    if correlation.isna().all().all():
        raise ValueError("Correlations could not be computed. The columns may not share rows "
                         "that have values.")
    count = correlation.shape[0]
    size = max(5.5, 0.62 * count + 3.0)
    figure, axis = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(correlation, annot=count <= 12, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=axis,
                cbar_kws={"label": "Pearson correlation"})
    axis.set_title(title or "Numeric correlation heatmap")
    try:
        figure.tight_layout()
    except Exception:
        pass
    table = correlation.round(4)
    table.index.name = "column"
    return figure, table.reset_index(), {
        "total_rows": int(len(data)), "valid_rows": int(frame.dropna().shape[0])}


def _pair_plot(data, columns, color, title, notes):
    columns = list(dict.fromkeys(columns or []))
    if len(columns) < 2:
        raise ValueError("Select at least two numeric columns for a pair plot.")
    if len(columns) > MAX_PAIR_COLUMNS:
        raise ValueError(f"A pair plot supports at most {MAX_PAIR_COLUMNS} columns. "
                         "Remove some columns.")
    if color is not None and color in columns:
        raise ValueError("The color column must be different from the plotted columns.")
    frame = pd.DataFrame({str(name): _numeric(data, name) for name in columns})
    names = list(frame.columns)
    valid = frame.notna().all(axis=1)
    if color:
        labels, color_valid = _labels(data, color)
        valid = valid & color_valid
    used = int(valid.sum())
    if used < 2:
        raise ValueError("Fewer than two rows have values in every selected column.")
    if used < len(data):
        notes.append(f"{len(data) - used:,} rows with a missing value in the selected "
                     "columns are left out.")
    frame = frame[valid].copy()
    palette = None
    hue = None
    if color:
        hue = f"{color} (group)"
        grouped, order, total = _top_groups(labels[valid])
        frame[hue] = grouped.to_numpy()
        palette = dict(zip(order, _colors(len(order))))
        if total:
            notes.append(f"'{color}' has {total:,} groups; the {MAX_COLOR_GROUPS} most common "
                         "are colored and the rest are grouped as 'Other'.")
    frame = _sample(frame, MAX_PAIR_ROWS, notes, "the pair plot")
    kwargs = {"plot_kws": {"s": 14, "alpha": 0.6, "edgecolor": "none"}}
    if hue is None:
        kwargs["plot_kws"]["color"] = PALETTE[0]
        kwargs["diag_kws"] = {"color": PALETTE[0]}
    else:
        kwargs["diag_kws"] = {"multiple": "stack"}
    grid = sns.pairplot(frame, vars=names, hue=hue, palette=palette, corner=True,
                        diag_kind="hist", height=2.1, **kwargs)
    figure = getattr(grid, "figure", None) or grid.fig
    figure.subplots_adjust(top=0.94)
    figure.suptitle(title or "Pair plot of numeric columns", y=0.99)
    return figure, frame.reset_index(drop=True), {
        "total_rows": int(len(data)), "valid_rows": used}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_custom_chart(
    data: pd.DataFrame,
    chart_type: str,
    grouping_column=None,
    value_column=None,
    aggregation: str = "Count",
    date_grouping: str = "month",
    *,
    color_column=None,
    size_column=None,
    columns=None,
    sort_order: str = "Value: high to low",
    top_n=None,
    bins: int = 20,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    """Build a custom graph from the actual current data.

    ``grouping_column`` is the X / category / group column and ``value_column`` the
    numeric Y column. The DataFrame is only read, never modified. Returns
    ``(figure, table)``; ``table`` holds the aggregated (or plotted) data and carries
    ``attrs["row_summary"]`` and ``attrs["notes"]``. Raises ``ValueError`` with a
    plain-language message when the chosen columns cannot make this chart.
    """

    if not isinstance(data, pd.DataFrame) or data.empty or data.shape[1] == 0:
        raise ValueError("The current dataset is empty.")
    spec = get_chart_spec(chart_type)
    x = _clean_optional(grouping_column)
    y = _clean_optional(value_column)
    color = _clean_optional(color_column)
    size = _clean_optional(size_column)
    title, x_label, y_label = (_clean_optional(item) for item in (title, x_label, y_label))
    aggregation = "Average" if aggregation == "Mean" else (aggregation or "Count")
    if spec["x"] in ("category", "numeric", "time") and x is None:
        raise ValueError(f"Select the {spec['x_text'].lower()}.")
    if spec["y"] == "numeric" and y is None:
        raise ValueError(f"Select the {spec['y_text'].lower()}.")
    if spec["color"] == "required" and color is None:
        raise ValueError(f"Select the {spec['color_text'].lower()}.")
    if spec["size"] and size is None:
        raise ValueError("Select a numeric column for the bubble size.")
    if not spec["y"]:
        y = None
    if not spec["color"]:
        color = None
    if spec["agg"]:
        if y is None and aggregation != "Count":
            raise ValueError(f"'{aggregation}' needs a numeric value column. "
                             "Select one, or use Count to count records.")
        if aggregation not in allowed_aggregations(chart_type, y is not None):
            raise ValueError(f"{aggregation} is not available for a {chart_type.lower()}. "
                             f"Use: {', '.join(allowed_aggregations(chart_type, y is not None))}.")
    else:
        aggregation = "Count"
    if top_n is not None:
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            raise ValueError("Top-N must be a whole number.")
        top_n = top_n if top_n >= 1 else None
    try:
        bins = max(2, min(200, int(bins)))
    except (TypeError, ValueError):
        raise ValueError("The number of bins must be a whole number.")
    notes: list[str] = []

    open_figures = set(plt.get_fignums())
    try:
        if chart_type in ("Bar chart", "Horizontal bar chart", "Pie chart", "Donut chart"):
            figure, table, summary = _category_chart(
                data, chart_type, x, y, aggregation, sort_order, top_n,
                title, x_label, y_label, notes)
        elif chart_type == "Stacked bar chart":
            figure, table, summary = _stacked_bar(
                data, x, y, color, aggregation, sort_order, top_n,
                title, x_label, y_label, notes)
        elif chart_type == "Histogram":
            figure, table, summary = _histogram(data, x, color, bins, title, x_label, y_label, notes)
        elif chart_type in ("Box plot", "Violin plot"):
            figure, table, summary = _box_or_violin(
                data, chart_type, x, y, sort_order, top_n, title, x_label, y_label, notes)
        elif chart_type in ("Scatter plot", "Bubble chart"):
            figure, table, summary = _scatter_or_bubble(
                data, chart_type, x, y, color, size, title, x_label, y_label, notes)
        elif chart_type in ("Line chart", "Area chart"):
            figure, table, summary = _time_chart(
                data, chart_type, x, y, color, aggregation, date_grouping,
                title, x_label, y_label, notes)
        elif chart_type == "Correlation heatmap":
            figure, table, summary = _heatmap(data, columns, title, notes)
        else:
            figure, table, summary = _pair_plot(data, columns, color, title, notes)
    except Exception:
        for number in set(plt.get_fignums()) - open_figures:
            plt.close(number)
        raise

    summary = dict(summary)
    summary["excluded_rows"] = max(0, summary["total_rows"] - summary["valid_rows"])
    table.attrs["row_summary"] = summary
    table.attrs["notes"] = notes
    return figure, table


# ---------------------------------------------------------------------------
# Automatic visualizations (unchanged)
# ---------------------------------------------------------------------------

def build_chart(data: pd.DataFrame, chart_name: str, results: Optional[dict] = None):
    """Build one supported chart, returning ``(figure, interpretation)``."""

    results = results or {}
    frame = results.get("results_dataframe")
    source = frame if isinstance(frame, pd.DataFrame) else data
    columns = source.columns
    figure, axis = plt.subplots(figsize=(8, 4.8))

    if chart_name == "sentiment":
        column = "predicted_sentiment"
        if column not in columns:
            plt.close(figure)
            return None
        values = source[column].dropna().astype(str).value_counts()
        axis.pie(values, labels=values.index, autopct="%1.1f%%",
                 colors=["#6d4fe6", "#f06a8b"], startangle=90)
        axis.set_title("Predicted sentiment")
        return figure, "Share of analyzed feedback predicted as positive or negative."

    if chart_name == "emotion":
        column = "predicted_emotion"
        if column not in columns:
            plt.close(figure)
            return None
        values = source[column].dropna().astype(str).value_counts().head(12)
        values.sort_values().plot.barh(ax=axis, color="#14b8a6")
        axis.set_title("Emotion distribution")
        axis.set_xlabel("Analyzed records")
        return figure, "Most common predicted emotions in analyzed feedback."

    if chart_name == "topics":
        column = "discovered_topic"
        if column not in columns:
            plt.close(figure)
            return None
        values = source[column].dropna().astype(str).value_counts()
        values.sort_values().plot.barh(ax=axis, color="#6d4fe6")
        axis.set_title("Discovered topic frequency")
        axis.set_xlabel("Analyzed records")
        return figure, "Topic counts are based on the NMF assignments returned by the model."

    if chart_name == "rating":
        ratings = detect_rating_columns(data)
        if not ratings:
            plt.close(figure)
            return None
        column = ratings[0]["column"]
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        values = values[values.between(1, 5)]
        if values.empty:
            plt.close(figure)
            return None
        values.value_counts().sort_index().plot.bar(ax=axis, color="#f0a34e")
        axis.set_title(f"Rating distribution — {column}")
        axis.set_xlabel("Rating")
        axis.set_ylabel("Records")
        return figure, "Ratings are shown only when values fit the detected 1–5 rating range."

    if chart_name == "keywords":
        counts = results.get("keyword_counts", {})
        if not counts:
            plt.close(figure)
            return None
        values = pd.Series(counts).sort_values().tail(15)
        values.plot.barh(ax=axis, color="#14b8a6")
        axis.set_title("Most frequent feedback terms")
        axis.set_xlabel("Occurrences")
        return figure, "Terms are counted from the selected feedback column after basic stop-word filtering."

    if chart_name == "trend":
        dates = detect_date_columns(data)
        if not dates:
            plt.close(figure)
            return None
        date_column = dates[0]["column"]
        trend = pd.DataFrame({"date": parse_dates(data[date_column])}).dropna()
        if trend.empty:
            plt.close(figure)
            return None
        counts = trend.assign(period=trend["date"].dt.to_period("M").astype(str))
        counts["period"].value_counts().sort_index().plot(ax=axis, marker="o", color="#6d4fe6")
        axis.set_title(f"Records over time — {date_column}")
        axis.set_xlabel("Month")
        axis.set_ylabel("Records")
        axis.tick_params(axis="x", rotation=35)
        return figure, "Monthly record volume based on the first detected date column."

    if chart_name == "numeric":
        numeric = data.select_dtypes(include="number").columns.tolist()
        if not numeric:
            plt.close(figure)
            return None
        data[numeric[0]].dropna().plot.hist(ax=axis, bins=12, color="#14b8a6")
        axis.set_title(f"Distribution — {numeric[0]}")
        axis.set_xlabel(str(numeric[0]))
        return figure, "Distribution of the first supported numeric column."

    if chart_name == "correlation":
        numeric = data.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            plt.close(figure)
            return None
        sns.heatmap(numeric.corr(), annot=True, cmap="vlag", center=0, ax=axis, fmt=".2f")
        axis.set_title("Numeric correlation heatmap")
        return figure, "Pearson correlations across supported numeric columns; correlation is not causation."

    plt.close(figure)
    return None


def supported_charts(data: pd.DataFrame, results: Optional[dict] = None) -> list[str]:
    """Return only visualizations supported by the current dataset."""

    results = results or {}
    frame = results.get("results_dataframe")
    source = frame if isinstance(frame, pd.DataFrame) else data
    available = []
    if "predicted_sentiment" in source:
        available.append("sentiment")
    if "predicted_emotion" in source:
        available.append("emotion")
    if "discovered_topic" in source:
        available.append("topics")
    if detect_rating_columns(data):
        available.append("rating")
    if detect_date_columns(data):
        available.append("trend")
    if data.select_dtypes(include="number").shape[1]:
        available.append("numeric")
    if data.select_dtypes(include="number").shape[1] >= 2:
        available.append("correlation")
    if results.get("keyword_counts"):
        available.append("keywords")
    return available