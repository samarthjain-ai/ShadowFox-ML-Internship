import re
import pandas as pd


def clean_text(text):
    """Clean an individual feedback text."""

    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_text_column(data, text_column):
    """Return a cleaned copy of the selected text column."""

    if text_column not in data.columns:
        raise ValueError(f"Column '{text_column}' does not exist.")

    result = data.copy()
    result[text_column] = result[text_column].apply(clean_text)

    return result