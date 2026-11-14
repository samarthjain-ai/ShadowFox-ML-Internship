import unittest

import pandas as pd

from src.assistant_engine import (
    answer_question,
    build_dataset_signature,
    generate_quick_insights,
    get_dataset_context,
    get_guided_questions,
    reset_assistant_context,
)


class AssistantEngineTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "review": [
                "Fast delivery and excellent support",
                "The product broke after one day",
                "Helpful team but slow delivery",
                "Excellent quality and friendly service",
            ],
            "rating": [5, 1, 3, 5],
            "category": ["delivery", "quality", "delivery", "service"],
            "amount": [10.0, 20.0, 15.0, 30.0],
        }, index=[10, 11, 12, 13])
        self.results = {
            "results_dataframe": pd.DataFrame({
                "review": self.data["review"],
                "predicted_sentiment": [
                    "positive", "negative", "negative", "positive"
                ],
                "predicted_emotion": ["joy", "anger", "sadness", "joy"],
                "discovered_topic": [1, 2, 1, 1],
            }, index=self.data.index),
            "analyses": {
                "topics": {
                    "status": "completed",
                    "topics": [
                        {"topic": 1, "keywords": ["delivery", "service"]},
                        {"topic": 2, "keywords": ["quality", "broken"]},
                    ],
                }
            },
        }
        self.context = get_dataset_context(
            self.data,
            self.results,
            text_column="review"
        )

    def test_dataset_overview_uses_original_data(self):
        response = answer_question("How many rows and columns are present?", self.context)
        self.assertEqual(response["evidence"]["rows"], 4)
        self.assertEqual(response["evidence"]["columns"], 4)
        self.assertIn("rating", response["evidence"]["column_types"]["numeric"])

    def test_sentiment_percentage_and_chart(self):
        response = answer_question(
            "What percentage of reviews are positive or negative?",
            self.context
        )
        self.assertEqual(response["intent"], "sentiment")
        self.assertEqual(response["evidence"]["percentages"]["positive"], 50.0)
        self.assertEqual(response["evidence"]["percentages"]["negative"], 50.0)
        self.assertEqual(response["charts"][0]["type"], "pie")

    def test_emotion_distribution(self):
        response = answer_question("Which emotions are most common?", self.context)
        self.assertEqual(response["evidence"]["counts"]["joy"], 2)
        self.assertEqual(response["charts"][0]["type"], "bar")

    def test_topics_and_keywords(self):
        topics = answer_question("What are the main discussion topics?", self.context)
        keywords = answer_question(
            "What are the most frequently occurring meaningful keywords?",
            self.context
        )
        self.assertEqual(topics["evidence"]["topics"][0]["topic"], 1)
        self.assertTrue(keywords["evidence"]["keywords"])

    def test_numeric_and_rating_statistics(self):
        numeric = answer_question("Calculate the average and median.", self.context)
        rating = answer_question("Show the rating distribution.", self.context)
        self.assertEqual(numeric["evidence"]["amount"]["mean"], 18.75)
        self.assertEqual(rating["evidence"]["minimum"], 1.0)
        self.assertEqual(rating["charts"][0]["type"], "bar")

    def test_missing_analysis_and_unsupported_question(self):
        context = get_dataset_context(self.data, {}, text_column="review")
        sentiment = answer_question("Show sentiment distribution.", context)
        unsupported = answer_question("What is tomorrow's weather?", context)
        self.assertFalse(sentiment["evidence"]["available"])
        self.assertEqual(unsupported["intent"], "unsupported")
        self.assertIn("not an unrestricted", unsupported["answer"].lower())

    def test_eda_generates_chart_specs(self):
        response = answer_question("Perform basic EDA.", self.context)
        self.assertEqual(response["intent"], "eda")
        self.assertTrue(response["charts"])
        self.assertTrue(response["evidence"]["numeric_statistics"])

    def test_dataset_signature_clears_stale_context(self):
        state = {
            "dataset_signature": "old",
            "analysis_results": {"old": True},
            "assistant_dataset": self.data,
            "assistant_metadata": {"sheet_name": "Old"},
            "assistant_history": [{"answer": "old"}],
            "assistant_question": "old question",
            "chat_minimized": True,
        }
        signature = build_dataset_signature(b"new", "reviews.csv", header=0)
        reset_assistant_context(state, signature)
        self.assertEqual(state["dataset_signature"], signature)
        self.assertNotIn("analysis_results", state)
        self.assertNotIn("assistant_history", state)
        self.assertNotIn("assistant_question", state)
        self.assertNotIn("chat_minimized", state)

    def test_guided_questions_reflect_available_capabilities(self):
        suggestions = get_guided_questions(self.context)
        self.assertIn("Dataset overview", suggestions)
        self.assertIn("Sentiment", suggestions)
        self.assertIn("Emotion", suggestions)
        self.assertIn("Topics and keywords", suggestions)
        self.assertIn("Ratings", suggestions)
        self.assertIn("Numerical analysis", suggestions)
        self.assertIn("EDA", suggestions)
        self.assertIn("Business insights", suggestions)

    def test_guided_questions_hide_missing_capabilities(self):
        data = pd.DataFrame({
            "feedback": ["A useful item", "A disappointing item"],
            "group": ["a", "b"],
        })
        context = get_dataset_context(data, {}, text_column="feedback")
        suggestions = get_guided_questions(context)
        self.assertIn("Dataset overview", suggestions)
        self.assertNotIn("Sentiment", suggestions)
        self.assertNotIn("Emotion", suggestions)
        self.assertNotIn("Topics and keywords", suggestions)
        self.assertNotIn("Ratings", suggestions)
        self.assertNotIn("Numerical analysis", suggestions)

    def test_quick_insights_use_completed_results(self):
        insights = generate_quick_insights(self.context)
        self.assertTrue(insights["available"])
        self.assertIn("sentiment", insights["evidence"])
        self.assertIn("emotion", insights["evidence"])
        self.assertTrue(insights["bullets"])

    def test_quick_insights_explain_when_model_analysis_is_missing(self):
        context = get_dataset_context(
            self.data[["review", "rating"]],
            {},
            text_column="review"
        )
        insights = generate_quick_insights(context)
        self.assertTrue(insights["available"])
        self.assertNotIn("sentiment", insights["evidence"])
        self.assertTrue(insights["bullets"])


if __name__ == "__main__":
    unittest.main()
