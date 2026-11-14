import re
import warnings

import pandas as pd

# Text values that mean "no value" in spreadsheets and CSV exports.
_MISSING_TOKENS = {
    "", "nan", "nat", "none", "null", "n/a", "na", "#n/a", "-", "--",
}
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}(?:\D|$)")
_DATE_SAMPLE_SIZE = 2000


def _blank_to_missing(series):
    """Return ``series`` as text with blank and "N/A"-style values set to NA."""

    text = series.astype("string").str.strip()
    return text.mask(text.str.lower().isin(_MISSING_TOKENS))


def _infer_dayfirst(text):
    """Decide whether numeric dates such as 03/04/2024 are day-first.

    A value like 25/03/2024 can only be day-first, and 03/25/2024 only
    month-first, so any unambiguous value settles the whole column. When every
    value is ambiguous (all days and months are 12 or less) pandas' month-first
    default is kept.
    """

    parts = text.dropna().str.extract(_NUMERIC_DATE)
    if parts.empty:
        return False
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    first_large = bool((first > 12).any())
    second_large = bool((second > 12).any())
    return first_large and not second_large


def parse_dates(series):
    """Parse a column into timezone-naive datetimes; invalid values become NaT.

    Blank cells and "N/A"-style text are treated as missing, day-first versus
    month-first numeric dates are inferred from the data, and timezones are
    dropped so grouping by month never shifts or warns.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    else:
        text = _blank_to_missing(series)
        dayfirst = _infer_dayfirst(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(
                text, errors="coerce", format="mixed", dayfirst=dayfirst,
            )
            if not pd.api.types.is_datetime64_any_dtype(parsed):
                # Mixed UTC offsets come back as objects; normalise to UTC.
                parsed = pd.to_datetime(
                    text, errors="coerce", format="mixed",
                    dayfirst=dayfirst, utc=True,
                )
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _value_masks(data):
    """Return masks for null, empty-string, and whitespace-only values."""

    null_mask = data.isna()
    string_values = data.astype("string")
    non_null = ~null_mask
    empty_mask = non_null & string_values.eq("")
    whitespace_mask = (
        non_null
        & ~empty_mask
        & string_values.apply(lambda column: column.str.strip().eq(""))
    )

    return null_mask, empty_mask, whitespace_mask


def profile_dataset(data):
    """Return null, blank, structural, and basic dataset information."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    null_mask, empty_mask, whitespace_mask = _value_masks(data)
    blank_mask = empty_mask | whitespace_mask
    empty_row_mask = (null_mask | blank_mask).all(axis=1)
    empty_column_mask = (null_mask | blank_mask).all(axis=0)

    profile = {
        "rows": len(data),
        "columns": len(data.columns),
        "column_names": data.columns.tolist(),
        "data_types": data.dtypes.astype(str).to_dict(),
        "missing_values": null_mask.sum().to_dict(),
        "missing_value_total": int(null_mask.sum().sum()),
        "empty_string_values": empty_mask.sum().to_dict(),
        "empty_string_total": int(empty_mask.sum().sum()),
        "whitespace_only_values": whitespace_mask.sum().to_dict(),
        "whitespace_only_total": int(whitespace_mask.sum().sum()),
        "blank_value_total": int(blank_mask.sum().sum()),
        "completely_empty_rows": int(empty_row_mask.sum()),
        "completely_empty_row_indices": data.index[
            empty_row_mask
        ].tolist(),
        "completely_empty_columns": data.columns[
            empty_column_mask
        ].tolist(),
        "unnamed_columns": [
            column for column in data.columns
            if str(column).strip().lower().startswith("unnamed:")
        ],
        "duplicate_rows": int(data.duplicated().sum()),
        "memory_usage_mb": round(
            data.memory_usage(deep=True).sum() / (1024 ** 2), 2
        )
    }

    return profile


def detect_text_columns(data, min_average_length=20):
    """Rank likely feedback columns using content and quality signals."""

    candidates = []
    row_count = max(len(data), 1)
    feedback_names = (
        "review", "feedback", "comment", "response", "description",
        "message", "complaint", "body", "text", "content"
    )

    for column in data.columns:
        if not (
            pd.api.types.is_object_dtype(data[column])
            or pd.api.types.is_string_dtype(data[column])
        ):
            continue

        values = data[column].dropna().astype(str).str.strip()
        values = values[values.ne("")]

        if values.empty:
            continue

        average_length = values.str.len().mean()
        unique_ratio = values.nunique() / max(len(values), 1)
        usable_ratio = len(values) / row_count
        meaningful_ratio = (
            values.str.len().ge(min_average_length)
            | values.str.contains(r"\s", regex=True)
        ).mean()
        sentence_ratio = values.str.contains(
            r"[.!?]|,\s",
            regex=True
        ).mean()
        normalized_name = str(column).strip().lower()
        name_signal = any(
            token in normalized_name
            for token in feedback_names
        )
        suspicious_name = normalized_name.startswith("unnamed:")

        length_signal = min(average_length / max(min_average_length, 1), 1)
        confidence = (
            0.25 * usable_ratio
            + 0.25 * length_signal
            + 0.20 * meaningful_ratio
            + 0.15 * sentence_ratio
            + 0.15 * float(name_signal)
            - 0.05 * float(suspicious_name)
        )

        reasons = []
        if name_signal:
            reasons.append("feedback-like column name")
        if suspicious_name:
            reasons.append("suspicious unnamed column name")
        if usable_ratio >= 0.8:
            reasons.append("high usable-row coverage")
        if meaningful_ratio >= 0.5:
            reasons.append("values resemble meaningful text")
        if sentence_ratio >= 0.25:
            reasons.append("contains sentence-like punctuation")

        has_text_signal = (
            meaningful_ratio > 0
            or sentence_ratio > 0
            or name_signal
        )

        if len(values) >= 2 and confidence >= 0.35 and has_text_signal:
            candidates.append({
                "column": column,
                "average_length": round(average_length, 2),
                "unique_ratio": round(unique_ratio, 3),
                "non_empty_count": int(len(values)),
                "usable_text_percentage": round(usable_ratio * 100, 2),
                "meaningful_text_percentage": round(
                    meaningful_ratio * 100, 2
                ),
                "sentence_like_percentage": round(
                    sentence_ratio * 100, 2
                ),
                "confidence": round(min(max(confidence, 0), 1), 3),
                "suspicious_name": suspicious_name,
                "reasons": reasons
            })

    return sorted(
        candidates,
        key=lambda item: (
            item["confidence"],
            item["average_length"]
        ),
        reverse=True
    )


def detect_label_columns(data):
    """Find columns containing binary sentiment/classification labels.

    Numeric star ratings are intentionally excluded; use
    :func:`detect_rating_columns` for those.
    """

    possible_names = {
        "sentiment", "label", "class", "target", "outcome",
        "ground_truth", "ground truth"
    }
    columns = []

    for column in data.columns:
        values = data[column].dropna()
        if values.empty:
            continue

        normalized = values.astype(str).str.strip().str.lower()
        normalized_set = set(normalized)
        name = str(column).strip().lower().replace("_", " ")
        name_match = name in possible_names
        binary_values = normalized_set.issubset({
            "positive", "negative", "true", "false", "0", "1"
        })

        if (name_match and binary_values) or (
            binary_values and len(normalized_set) == 2
        ):
            columns.append(column)

    return columns


def detect_rating_columns(data, minimum_valid_ratio=0.9):
    """Find numeric or star-rating columns separately from labels.

    A column qualifies when it is named like a rating and at least
    ``minimum_valid_ratio`` of its numeric values lie in the 1-5 range. A few
    invalid values (a stray 0 or 99) no longer disqualify the whole column;
    they are counted in ``out_of_range_count`` and excluded from charts.
    """

    columns = []

    for column in data.columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue

        name = str(column).strip().lower().replace("_", " ")
        name_match = (
            any(token in name for token in ("rating", "score"))
            or re.search(r"\bstars?\b", name) is not None
        )
        in_range = values.between(1, 5)
        valid_ratio = float(in_range.mean())

        if name_match and in_range.any() and valid_ratio >= minimum_valid_ratio:
            valid_values = values[in_range]
            columns.append({
                "column": column,
                "minimum": float(valid_values.min()),
                "maximum": float(valid_values.max()),
                "type": "star_rating",
                "out_of_range_count": int((~in_range).sum()),
            })

    return columns


def detect_date_columns(data, minimum_parseable_ratio=0.8):
    """Return columns whose non-blank values consistently parse as dates.

    Blank cells and "N/A"-style text do not count against a column. Long
    columns are judged on a fixed-size sample so detection stays fast.
    """

    detected = []
    for column in data.columns:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            if series.notna().any():
                detected.append({"column": column, "parseable_percentage": 100.0})
            continue
        text = _blank_to_missing(series).dropna()
        if text.empty:
            continue
        if len(text) > _DATE_SAMPLE_SIZE:
            text = text.sample(_DATE_SAMPLE_SIZE, random_state=0)
        ratio = float(parse_dates(text).notna().mean())
        name = str(column).strip().lower().replace("_", " ")
        name_match = any(token in name for token in ("date", "time", "month", "year"))
        if ratio >= minimum_parseable_ratio and (name_match or ratio == 1):
            detected.append({
                "column": column,
                "parseable_percentage": round(ratio * 100, 2),
            })
    return detected


def explain_date_rating_detection(data, max_columns=40):
    """Explain, column by column, why a column was or was not detected.

    Returns a DataFrame with the same rules as :func:`detect_date_columns` and
    :func:`detect_rating_columns`, so a user can see exactly what to change
    when the Custom Graph Builder cannot find a Date or Rating column.
    """

    rows = []
    for column in list(data.columns)[:max_columns]:
        series = data[column]
        name = str(column).strip().lower().replace("_", " ")

        if pd.api.types.is_datetime64_any_dtype(series):
            date_check = "OK (date/time column)" if series.notna().any() else "Column is empty"
        elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            date_check = "Not a date: values are numbers"
        else:
            text = _blank_to_missing(series).dropna()
            if text.empty:
                date_check = "Column is empty"
            else:
                if len(text) > _DATE_SAMPLE_SIZE:
                    text = text.sample(_DATE_SAMPLE_SIZE, random_state=0)
                ratio = float(parse_dates(text).notna().mean())
                name_match = any(
                    token in name for token in ("date", "time", "month", "year")
                )
                if ratio < 0.8:
                    date_check = f"No: only {ratio:.0%} of values read as dates (needs 80%)"
                elif name_match or ratio == 1:
                    date_check = "OK"
                else:
                    date_check = (
                        "No: name should include date, time, month, or year "
                        "unless every value is a date"
                    )

        numbers = pd.to_numeric(series, errors="coerce").dropna()
        if numbers.empty:
            rating_check = "Not a rating: no numeric values"
        else:
            name_match = (
                any(token in name for token in ("rating", "score"))
                or re.search(r"\bstars?\b", name) is not None
            )
            in_range = float(numbers.between(1, 5).mean())
            if not name_match:
                rating_check = "No: name should include rating, score, or stars"
            elif in_range < 0.9:
                rating_check = f"No: only {in_range:.0%} of values are between 1 and 5 (needs 90%)"
            else:
                rating_check = "OK"

        rows.append({
            "Column": str(column),
            "Type": str(series.dtype),
            "Usable as Date?": date_check,
            "Usable as Rating?": rating_check,
        })
    return pd.DataFrame(rows)


def detect_categorical_columns(data, max_unique=50):
    """Return low-cardinality categorical columns suitable for comparisons."""

    columns = []
    for column in data.columns:
        series = data[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        unique_count = int(series.astype(str).nunique())
        if unique_count <= max_unique and unique_count < len(series):
            columns.append({
                "column": column,
                "unique_values": unique_count,
            })
    return columns


def validate_feedback_column(data, column):
    """Validate that a selected feedback column contains usable text."""

    if column not in data.columns:
        raise ValueError(f"Feedback column '{column}' was not found.")
    values = data[column].dropna().astype(str).str.strip()
    usable = values[values.ne("")]
    if usable.empty:
        raise ValueError(
            f"'{column}' does not contain any non-empty feedback text. "
            "Choose a text column with customer comments."
        )
    if (
        pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
        or pd.api.types.is_datetime64_any_dtype(data[column])
    ):
        raise ValueError(
            f"'{column}' is not a text column and is unsuitable for feedback."
        )
    average_length = float(usable.str.len().mean())
    if average_length < 5:
        raise ValueError(
            f"'{column}' contains very short values and is unlikely to be "
            "customer feedback. Choose a review or comment column."
        )
    return {
        "column": column,
        "usable_rows": int(len(usable)),
        "coverage_percentage": round(len(usable) / max(len(data), 1) * 100, 2),
        "average_length": round(average_length, 2),
    }