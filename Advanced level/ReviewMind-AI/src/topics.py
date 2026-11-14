
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def discover_topics(
    texts,
    n_topics=5,
    n_words=8,
    max_features=5000
):
    """Discover themes using TF-IDF and NMF."""

    if texts is None:
        raise ValueError("Text input cannot be None.")

    texts = [
        str(text).strip()
        for text in texts
        if text is not None and str(text).strip()
    ]

    if len(texts) < 2:
        raise ValueError(
            "At least two non-empty text records are required."
        )

    if n_topics < 1 or n_words < 1 or max_features < 1:
        raise ValueError(
            "n_topics, n_words and max_features must be positive."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=2,
        max_df=0.95
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        raise ValueError(
            "Topic discovery failed. The feedback may be too short, "
            "repetitive, or contain no usable vocabulary."
        ) from error

    if matrix.shape[1] == 0:
        raise ValueError(
            "No usable terms were found for topic discovery."
        )

    actual_topics = min(
        n_topics,
        matrix.shape[0],
        matrix.shape[1]
    )

    model = NMF(
        n_components=actual_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )

    document_topics = model.fit_transform(matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for topic_index, weights in enumerate(model.components_):

        top_indices = weights.argsort()[::-1][:n_words]

        topics.append({
            "topic": topic_index + 1,
            "keywords": feature_names[top_indices].tolist()
        })

    assignments = document_topics.argmax(axis=1) + 1

    return {
        "topics": topics,
        "document_topic": assignments.tolist(),
        "model": model,
        "vectorizer": vectorizer
    }
