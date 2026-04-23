"""Recall-first metrics: recall, F1, ROC-AUC, accuracy + confusion matrix."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    # === Metrik öncelik sırası ===
    # Recall en üstte: sağlık probleminde bir hastayı kaçırmanın (FN)
    # maliyeti, sağlıklıya yanlış alarm vermekten (FP) çok daha yüksek.
    # F1 ikinci sırada (precision-recall dengesi), ROC-AUC üçüncü
    # (eşikten bağımsız ayırıcılık), accuracy son sırada (dengesizlik
    # olunca yanıltıcı olabilir).
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    # ROC-AUC opsiyonel: sadece olasılık skoru verildiyse hesaplanır.
    # predict() yalnız 0/1 döndüğü için tek başına AUC hesabına yetmez.
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # === Confusion matrix — sayıları okumak metriklerden daha öğretici ===
    # Sıkıştırılmış 4 değer: TN, FP, FN, TP. Sunumda jüriye "kaç hastayı
    # kaçırdık?" sorusuna cevap vermek için FN değerini göstermek yeterli.
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # === Konsol raporu ===
    # main.py'de her aşamadan sonra bu blok yazdırılır. Format her model
    # için aynı olduğundan ekranda hızlı karşılaştırma yapılabilir.
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
