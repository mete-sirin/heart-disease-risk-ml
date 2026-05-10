from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} ===")
    print(f"Recall:    {metrics['recall']:.4f}  (priority: minimize FN)")
    print(f"F1-score:  {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Confusion matrix:")
    print(f"  TN={tn:3d}  FP={fp:3d}")
    print(f"  FN={fn:3d}  TP={tp:3d}")

    return metrics
