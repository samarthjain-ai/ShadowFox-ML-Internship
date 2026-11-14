
from functools import lru_cache
from pathlib import Path

from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMOTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "emotion_english_distilroberta"
)


@lru_cache(maxsize=1)
def load_emotion_model():
    """Load and cache the locally stored emotion model."""

    if not EMOTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Emotion model not found at: {EMOTION_MODEL_PATH}"
        )

    return pipeline(
        "text-classification",
        model=str(EMOTION_MODEL_PATH),
        tokenizer=str(EMOTION_MODEL_PATH),
        top_k=1,
        device=-1
    )


def predict_emotions(
    texts,
    batch_size=16
):
    """Predict the most likely emotion for each text."""

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

    emotion_model = load_emotion_model()

    results = emotion_model(
        texts,
        batch_size=batch_size,
        truncation=True
    )

    predictions = []

    for result in results:

        best = result[0] if isinstance(result, list) else result

        predictions.append({
            "predicted_emotion": best["label"],
            "confidence": round(float(best["score"]), 4)
        })

    return predictions
