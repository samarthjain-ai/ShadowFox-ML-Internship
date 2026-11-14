
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_tfidf_model(model_path):
    """Load the saved TF-IDF vectorizer and Logistic Regression model."""

    artifact = joblib.load(model_path)

    if "vectorizer" not in artifact or "model" not in artifact:
        raise ValueError(
            "Invalid TF-IDF model artifact. "
            "Expected vectorizer and model."
        )

    return artifact["vectorizer"], artifact["model"]


@lru_cache(maxsize=2)
def load_distilbert_model(model_path):
    """Load a fine-tuned DistilBERT model and tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(
    texts,
    model_type="tfidf",
    model_path=None,
    batch_size=16
):
    """Predict positive or negative sentiment for input texts."""

    if texts is None:
        return []

    if isinstance(texts, str):
        texts = [texts]

    texts = [
        str(text).strip()
        for text in texts
        if text is not None and str(text).strip()
    ]

    if not texts:
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if model_type not in {"tfidf", "distilbert"}:
        raise ValueError(
            "model_type must be 'tfidf' or 'distilbert'."
        )

    if model_path is None:

        if model_type == "tfidf":
            model_path = (
                PROJECT_ROOT
                / "models"
                / "tfidf_logistic_regression.pkl"
            )

        else:
            model_path = (
                PROJECT_ROOT
                / "models"
                / "distilbert_imdb_sentiment"
            )

    model_path = str(Path(model_path).resolve())

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    predictions = []
    confidence = []

    # TF-IDF + LOGISTIC REGRESSION
    if model_type == "tfidf":

        vectorizer, model = load_tfidf_model(model_path)

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            features = vectorizer.transform(batch_texts)

            batch_predictions = model.predict(features)
            probabilities = model.predict_proba(features)

            predictions.extend(batch_predictions)
            confidence.extend(probabilities.max(axis=1))

    # DISTILBERT
    elif model_type == "distilbert":

        tokenizer, model, device = load_distilbert_model(
            model_path
        )

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=192,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            with torch.inference_mode():

                logits = model(**encoded).logits

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

            predictions.extend(
                probabilities.argmax(dim=1).cpu().numpy()
            )

            confidence.extend(
                probabilities.max(dim=1).values.cpu().numpy()
            )

    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence)

    labels = np.where(
        predictions == 1,
        "positive",
        "negative"
    )

    return [
        {
            "predicted_sentiment": str(label),
            "confidence": round(float(score), 4)
        }
        for label, score in zip(labels, confidence)
    ]
