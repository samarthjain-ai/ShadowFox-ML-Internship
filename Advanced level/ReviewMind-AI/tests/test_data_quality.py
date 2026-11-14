import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis_engine import run_analysis
from src.data_loader import inspect_workbook, load_dataset
from src.data_profiler import (
    detect_date_columns,
    detect_label_columns,
    detect_rating_columns,
    detect_text_columns,
    profile_dataset,
    validate_feedback_column,
)


class DataQualityTests(unittest.TestCase):
    def test_normal_csv_has_no_missing_values(self):
        data = pd.DataFrame({
            "review": ["Great product.", "Works exactly as expected."],
            "sentiment": ["positive", "positive"]
        })

        profile = profile_dataset(data)

        self.assertEqual(profile["missing_value_total"], 0)
        self.assertEqual(profile["blank_value_total"], 0)
        self.assertEqual(detect_label_columns(data), ["sentiment"])

    def test_excel_blank_rows_are_reported_and_preserved(self):
        data = pd.DataFrame({
            "review": ["Useful item.", None, "Another useful item."],
            "score": [5, None, 4]
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank_rows.xlsx"
            data.to_excel(path, index=False)
            loaded = load_dataset(path)

        self.assertEqual(len(loaded), 3)
        self.assertEqual(profile_dataset(loaded)["completely_empty_rows"], 1)

    def test_excel_unnamed_column_is_flagged_but_can_be_feedback(self):
        data = pd.DataFrame({
            "Unnamed: 1": [
                "The delivery was quick and the packaging was excellent.",
                "Customer support resolved my issue within one day."
            ],
            "category": ["delivery", "support"]
        })

        candidates = detect_text_columns(data)

        self.assertEqual(candidates[0]["column"], "Unnamed: 1")
        self.assertTrue(candidates[0]["suspicious_name"])

    def test_multiple_excel_sheets_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiple_sheets.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"review": ["first sheet"]}).to_excel(
                    writer,
                    sheet_name="Reviews",
                    index=False
                )
                pd.DataFrame({"value": [1]}).to_excel(
                    writer,
                    sheet_name="Metadata",
                    index=False
                )

            inspection = inspect_workbook(path)

            with self.assertRaisesRegex(ValueError, "Reviews, Metadata"):
                load_dataset(path)

            selected = load_dataset(path, sheet_name="Reviews")

        self.assertEqual(inspection["sheet_names"], ["Reviews", "Metadata"])
        self.assertEqual(selected.iloc[0, 0], "first sheet")

    def test_excel_leading_title_row_requires_header_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "title_row.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame([
                    ["Customer feedback export", None],
                    ["review", "rating"],
                    ["Useful product.", 5]
                ]).to_excel(
                    writer,
                    header=False,
                    index=False
                )

            inspection = inspect_workbook(path)

            with self.assertRaisesRegex(ValueError, "header=1"):
                load_dataset(path)

            selected = load_dataset(path, header=1)

        self.assertEqual(
            inspection["sheets"][0]["suggested_header_row"],
            1
        )
        self.assertEqual(selected.columns.tolist(), ["review", "rating"])

    def test_blank_and_missing_feedback_are_accounted_separately(self):
        data = pd.DataFrame({
            "review": [
                "A usable customer review.",
                "   ",
                None,
                "<br>"
            ],
            "other": ["x", None, None, None]
        })

        result = run_analysis(data, "review", analyses=[])
        accounting = result["row_accounting"]

        self.assertEqual(accounting["original_rows"], 4)
        self.assertEqual(accounting["rows_with_missing_feedback"], 1)
        self.assertEqual(accounting["rows_with_blank_feedback"], 2)
        self.assertEqual(accounting["rows_analyzed"], 1)
        self.assertEqual(accounting["rows_excluded"], 3)
        self.assertEqual(len(accounting["exclusion_reasons"]), 3)

    def test_ratings_are_not_binary_sentiment_labels(self):
        data = pd.DataFrame({
            "review": ["The product is sturdy and useful."] * 3,
            "rating": [1, 3, 5],
            "sentiment": ["positive", "negative", "positive"]
        })

        self.assertEqual(detect_label_columns(data), ["sentiment"])
        self.assertEqual(
            detect_rating_columns(data)[0]["column"],
            "rating"
        )

    def test_unrelated_text_columns_do_not_outrank_feedback(self):
        data = pd.DataFrame({
            "review": [
                "The product arrived early and works very well.",
                "Support answered my question and fixed the issue."
            ],
            "department": ["Operations", "Operations"],
            "identifier": ["ABC-001", "ABC-002"]
        })

        candidates = detect_text_columns(data)

        self.assertEqual(candidates[0]["column"], "review")
        self.assertNotIn(
            "department",
            [candidate["column"] for candidate in candidates]
        )

    def test_feedback_validation_rejects_numeric_columns(self):
        data = pd.DataFrame({"rating": [1, 2], "review": ["Good", "Bad"]})
        with self.assertRaisesRegex(ValueError, "not a text column"):
            validate_feedback_column(data, "rating")

    def test_feedback_validation_rejects_identifier_like_text(self):
        data = pd.DataFrame({"code": ["A1", "B2"], "review": ["Good product", "Slow delivery"]})
        with self.assertRaisesRegex(ValueError, "very short"):
            validate_feedback_column(data, "code")

    def test_date_columns_are_detected_only_when_parseable(self):
        data = pd.DataFrame({
            "created_at": ["2025-01-01", "2025-02-01", None],
            "notes": ["one", "two", "three"],
        })
        detected = detect_date_columns(data)
        self.assertEqual(detected[0]["column"], "created_at")


if __name__ == "__main__":
    unittest.main()
