import unittest

import matplotlib.pyplot as plt
import pandas as pd

from src.exporter import build_report_pdf
from src.visualization import (
    aggregate_date_values,
    build_chart,
    build_custom_chart,
    default_date_grouping,
    supported_charts,
)


class VisualizationAndExportTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "review": ["Great delivery", "Poor support"],
            "rating": [5, 1],
            "date": ["2025-01-01", "2025-02-01"],
        })
        self.results = {
            "results_dataframe": pd.DataFrame({
                "review": self.data["review"],
                "predicted_sentiment": ["positive", "negative"],
                "predicted_emotion": ["joy", "anger"],
                "discovered_topic": [1, 2],
            }),
            "analysis_summary": {"sentiment": "completed"},
            "row_count_analyzed": 2,
        }

    def test_supported_charts_are_dataset_aware(self):
        charts = supported_charts(self.data, self.results)
        self.assertIn("sentiment", charts)
        self.assertIn("rating", charts)
        self.assertIn("trend", charts)
        self.assertNotIn("correlation", charts)

    def test_chart_builder_returns_figure_for_supported_chart(self):
        built = build_chart(self.data, "sentiment", self.results)
        self.assertIsNotNone(built)
        plt.close(built[0])

    def test_pdf_report_has_pdf_signature(self):
        report = build_report_pdf(self.data, self.results)
        self.assertTrue(report.startswith(b"%PDF"))

    def test_custom_date_rating_average_groups_actual_values(self):
        data = pd.DataFrame({
            "date": ["2025-01-02", "2025-01-15", "2025-02-01", "invalid"],
            "rating": [5, 3, 1, "not-a-rating"],
        })
        aggregated = aggregate_date_values(data, "date", "rating", "month")
        self.assertEqual(aggregated["date_group"].tolist(), ["2025-01", "2025-02"])
        self.assertEqual(aggregated["average_rating"].tolist(), [4.0, 1.0])
        self.assertEqual(aggregated["record_count"].tolist(), [2, 1])

    def test_custom_pie_chart_returns_aggregated_data(self):
        figure, aggregated = build_custom_chart(
            self.data,
            chart_type="Pie chart",
            grouping_column="date",
            value_column="rating",
            aggregation="Average",
            date_grouping="month",
        )
        self.assertEqual(len(aggregated), 2)
        plt.close(figure)

    def test_custom_graph_defaults_to_month_for_many_dates(self):
        data = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=20),
            "rating": [1, 2, 3, 4, 5] * 4,
        })
        self.assertEqual(default_date_grouping(data, "date"), "month")

    def test_custom_graph_rejects_missing_valid_pairs(self):
        data = pd.DataFrame({"date": ["invalid"], "rating": ["bad"]})
        with self.assertRaisesRegex(ValueError, "No rows contain"):
            aggregate_date_values(data, "date", "rating")

    def test_custom_graph_excludes_out_of_range_ratings(self):
        data = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "rating": [5, 7],
        })
        aggregated = aggregate_date_values(data, "date", "rating", "month")
        self.assertEqual(aggregated["record_count"].tolist(), [1])
        self.assertEqual(aggregated["average_rating"].tolist(), [5.0])


if __name__ == "__main__":
    unittest.main()
