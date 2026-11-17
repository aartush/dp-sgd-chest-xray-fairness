"""Metrics: per-class recall, aggregation that keeps minority classes visible.

Accuracy is not reported as a headline because every dataset here is imbalanced;
a model that always predicts the majority class scores 74% on PneumoniaMNIST
while achieving 0% recall on the minority class. Macro-averaged recall and the
minority-class recall are the primary endpoints.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_recall_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    t = y_true.astype(int)
    p = y_pred.astype(int)
    out = {}
    for c in (0, 1):
        mask = t == c
        out[f"recall_class{c}"] = float((p[mask] == c).mean()) if mask.any() else float("nan")
    out["minority_recall"] = out["recall_class0"]
    out["majority_recall"] = out["recall_class1"]
    out["macro_recall"] = float(np.nanmean([out["recall_class0"], out["recall_class1"]]))
    out["gap_majority_minus_minority"] = out["majority_recall"] - out["minority_recall"]
    out["accuracy"] = float((t == p).mean())
    return out


def per_class_precision_binary(y_true, y_pred):
    t = y_true.astype(int)
    p = y_pred.astype(int)
    out = {}
    for c in (0, 1):
        sel = p == c
        out[f"precision_class{c}"] = float((t[sel] == c).mean()) if sel.any() else float("nan")
    return out


def binary_report(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = y_true.astype(int)
    probs = probs.astype(float)
    y_pred = (probs >= threshold).astype(int)
    out = per_class_recall_binary(y_true, y_pred)
    out.update(per_class_precision_binary(y_true, y_pred))
    # positive class = 0 (minority, "normal") to align with recall reporting
    try:
        out["auc_minority"] = float(roc_auc_score((y_true == 0).astype(int), -probs))
        out["auc_positive_class"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        out["auc_minority"] = float("nan")
        out["auc_positive_class"] = float("nan")
    return out


def multilabel_report(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = y_true.astype(int)
    probs = probs.astype(float)
    n_labels = y_true.shape[1]
    per_label_auc, per_label_recall, per_label_prec = [], [], []
    for j in range(n_labels):
        t, p = y_true[:, j], probs[:, j]
        if len(np.unique(t)) > 1:
            per_label_auc.append(float(roc_auc_score(t, p)))
        else:
            per_label_auc.append(float("nan"))
        pred = (p >= threshold).astype(int)
        pos = t == 1
        per_label_recall.append(float((pred[pos] == 1).mean()) if pos.any() else float("nan"))
        sel = pred == 1
        per_label_prec.append(float((t[sel] == 1).mean()) if sel.any() else float("nan"))

    aucs = np.array(per_label_auc, dtype=float)
    recs = np.array(per_label_recall, dtype=float)
    precs = np.array(per_label_prec, dtype=float)
    out = {
        "macro_auc": float(np.nanmean(aucs)),
        "macro_recall": float(np.nanmean(recs)),
        "macro_precision": float(np.nanmean(precs)),
        "min_label_recall": float(np.nanmin(recs)),
        "max_label_recall": float(np.nanmax(recs)),
    }
    for j in range(n_labels):
        out[f"auc_label{j}"] = per_label_auc[j]
        out[f"recall_label{j}"] = per_label_recall[j]
    return out


def worst_group_recall(binary: dict) -> float:
    return float(min(binary["recall_class0"], binary["recall_class1"]))
