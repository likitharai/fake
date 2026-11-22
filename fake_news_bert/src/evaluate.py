"""Evaluation metrics. Positive class is FAKE (the detection target)."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import LABEL_MAP

POSITIVE_LABEL = 0  # FAKE


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=POSITIVE_LABEL, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=[LABEL_MAP[0], LABEL_MAP[1]],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_class": LABEL_MAP[POSITIVE_LABEL],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": int(len(y_true)),
    }


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        obj, indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    ))
