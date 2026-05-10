from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import tensorflow as tf

from src import eda
from src.baseline import train_baseline
from src.evaluate import evaluate
from src.model import DECISION_THRESHOLD, build_model, train_model
from src.preprocess import RANDOM_STATE, get_splits



RESULTS_PATH = Path("outputs/results.json")


def banner(text: str) -> None:

    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def run_baseline_stage(X_train, y_train, X_test, y_test) -> dict:
    # ============================================================
    # LOJİSTİK REGRESYON — Referans (baseline) modeli
    # ------------------------------------------------------------
    # Doğrusal bir sınıflandırıcı; ANN'in geçmesi gereken çıtayı
    # belirler. Akış:
    #   1) train_baseline ile modeli eğit.
    #   2) predict       -> 0/1 sınıf tahmini (varsayılan 0.5 eşiği).
    #   3) predict_proba -> pozitif sınıfın olasılığı (ROC/AUC için).
    #   4) evaluate      -> recall, F1, AUC, accuracy hesapla & yazdır.
    # Recall öncelikli metriktir: kalp hastasını "sağlıklı" demek
    # (false negative) en pahalı hatadır.
    # ============================================================
    model = train_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return evaluate("Logistic Regression (baseline)", y_test, y_pred, y_proba)


def run_ann_stage(name, hidden, X_train, y_train, X_test, y_test) -> dict:
    # ============================================================
    # YAPAY SİNİR AĞI (ANN) — v1 ve v2 için ortak akış
    # ------------------------------------------------------------
    # Her çağrının BAŞINDA tohum (seed) sıfırlanır; böylece v1 ve v2
    # birbirini etkilemez ve sonuçlar tekrar üretilebilir kalır.
    #
    # Adımlar:
    #   1) build_model : girdi boyutuna göre Keras modelini kur
    #                    (Dense + Dropout 0.3, sigmoid çıkış).
    #   2) train_model : EarlyStopping ile eğitim; epoch sayısı
    #                    erken durmaya göre değişir, bu yüzden
    #                    history'den okunup loglanır.
    #   3) Tahmin      : sigmoid olasılığı DECISION_THRESHOLD (0.5)
    #                    ile 0/1 sınıfına çevrilir.
    #   4) evaluate    : metrikler hesaplanır, epoch bilgisi eklenir.
    # ============================================================
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = build_model(n_features=X_train.shape[1], hidden_units=hidden)
    history = train_model(model, X_train, y_train, verbose=0)
    epochs_run = len(history.history["loss"])
    print(f"[{name}] trained for {epochs_run} epochs")

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    metrics = evaluate(name, y_test, y_pred, y_proba)
    metrics["epochs"] = epochs_run
    return metrics


def print_summary(results: dict) -> None:
    banner("SUMMARY")
    header = f"{'Model':<36}{'Recall':>9}{'F1':>9}{'AUC':>9}{'Acc':>9}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<36}"
            f"{m['recall']:>9.4f}"
            f"{m['f1']:>9.4f}"
            f"{m.get('roc_auc', float('nan')):>9.4f}"
            f"{m['accuracy']:>9.4f}"
        )


def main() -> None:
    banner("STAGE 1: Exploratory Data Analysis")
    eda.run()

    banner("STAGE 2: Preprocessing & stratified train/test split")
    X_train, X_test, y_train, y_test, _ = get_splits()
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"y_test  positive rate: {y_test.mean():.3f}")

    banner("STAGE 3: Baseline (Logistic Regression)")
    m_base = run_baseline_stage(X_train, y_train, X_test, y_test)

    banner("STAGE 4: ANN v1 (16 -> 8, Dropout 0.3)")
    m_v1 = run_ann_stage(
        "ANN v1 (16->8)", (16, 8), X_train, y_train, X_test, y_test
    )

    banner("STAGE 5: ANN v2 (8, Dropout 0.3)")
    m_v2 = run_ann_stage("ANN v2 (8)", (8,), X_train, y_train, X_test, y_test)

    results = {
        "Logistic Regression": m_base,
        "ANN v1 (16->8)": m_v1,
        "ANN v2 (8)": m_v2,
    }
    print_summary(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
