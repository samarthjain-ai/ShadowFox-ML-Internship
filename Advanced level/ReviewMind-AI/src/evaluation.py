
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_classification(y_true, y_pred):
    """Calculate classification metrics for compatible labels."""

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if any(label is None for label in y_true + y_pred):
        raise ValueError("Labels cannot contain missing values.")

    # Convert labels to strings for consistent comparisons.
    y_true = [str(label).strip().lower() for label in y_true]
    y_pred = [str(label).strip().lower() for label in y_pred]

    if any(not label for label in y_true + y_pred):
        raise ValueError("Labels cannot contain empty values.")

    labels = sorted(set(y_true) | set(y_pred))

    if len(labels) < 2:
        raise ValueError(
            "At least two distinct classes are required for evaluation."
        )

    average = "binary" if len(labels) == 2 else "weighted"

    positive_label = (
        "positive" if "positive" in labels else labels[-1]
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            average=average,
            pos_label=positive_label,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average=average,
            pos_label=positive_label,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            average=average,
            pos_label=positive_label,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        ).tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True
        )
    }
