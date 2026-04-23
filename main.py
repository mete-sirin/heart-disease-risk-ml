"""End-to-end pipeline: EDA -> preprocess -> baseline -> ANN (v1, v2) -> summary."""

from __future__ import annotations

import os

# TF log seviyesini düşür — sunum sırasında konsol gürültüsünü azaltmak için.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import tensorflow as tf

from src import eda
from src.baseline import train_baseline
from src.evaluate import evaluate
from src.model import DECISION_THRESHOLD, build_model, train_model
from src.preprocess import RANDOM_STATE, get_splits


# === Sonuç dosyasının yolu ===
# Tüm modellerin metriklerini JSON olarak kaydediyoruz. Bu dosya
# rapor/sunum için ham referans noktası olur — sayıları "uydurmuyoruz".
RESULTS_PATH = Path("outputs/results.json")


def banner(text: str) -> None:
    # Konsolu aşamalara böler — sunum sırasında jüri "şu an hangi adımdayız?"
    # diye bakıp anlayabilsin diye görsel ayraç.
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def run_baseline_stage(X_train, y_train, X_test, y_test) -> dict:
    # === Lojistik regresyon — referans modelin tek satırlık akışı ===
    # Eğit, sınıf tahmini al, olasılık tahmini al, metrikleri yazdır.
    model = train_baseline(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return evaluate("Logistic Regression (baseline)", y_test, y_pred, y_proba)


def run_ann_stage(name, hidden, X_train, y_train, X_test, y_test) -> dict:
    # === ANN aşaması — v1 ve v2 için ortak akış ===
    # Her aşama BAŞINDA seed sıfırlanıyor; iki çalıştırma birbirini
    # etkilemesin ve sonuçlar tekrar üretilebilir kalsın diye.
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = build_model(n_features=X_train.shape[1], hidden_units=hidden)
    history = train_model(model, X_train, y_train, verbose=0)
    epochs_run = len(history.history["loss"])
    print(f"[{name}] trained for {epochs_run} epochs")

    # Sigmoid çıktıyı 0.5 eşiğiyle sınıfa dönüştürüyoruz.
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    metrics = evaluate(name, y_test, y_pred, y_proba)
    metrics["epochs"] = epochs_run
    return metrics


def print_summary(results: dict) -> None:
    # === Tek bakışta karşılaştırma tablosu ===
    # Sunumun en yüksek değerli ekranı: 3 modelin 4 metriği yan yana.
    # Burada hangi modelin kazandığı doğrudan görülüyor.
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
    # === STAGE 1: Keşifsel Veri Analizi ===
    # Konsola özet yazar, outputs/eda/ altına 4 figür kaydeder.
    banner("STAGE 1: Exploratory Data Analysis")
    eda.run()

    # === STAGE 2: Ön işleme + train/test split ===
    # Stratifiye 80/20 bölünme. Çıktıdaki pozitif oranlarının yakın olması
    # stratify'ın çalıştığının kanıtı.
    banner("STAGE 2: Preprocessing & stratified train/test split")
    X_train, X_test, y_train, y_test, _ = get_splits()
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"y_test  positive rate: {y_test.mean():.3f}")

    # === STAGE 3: Baseline ===
    # Lojistik regresyon — ANN'in geçmesi gereken çıta.
    banner("STAGE 3: Baseline (Logistic Regression)")
    m_base = run_baseline_stage(X_train, y_train, X_test, y_test)

    # === STAGE 4: ANN v1 — şartname mimarisi ===
    # Spec'teki iki katmanlı yapı. Sonuç: baseline'ın altında kaldı.
    banner("STAGE 4: ANN v1 (16 -> 8, Dropout 0.3)")
    m_v1 = run_ann_stage(
        "ANN v1 (16->8)", (16, 8), X_train, y_train, X_test, y_test
    )

    # === STAGE 5: ANN v2 — küçültülmüş mimari ===
    # Tek gizli katman. Hipotez: küçük veri için kapasite fazlaydı.
    # Sonuç: recall'da hem v1'i hem baseline'ı geçti.
    banner("STAGE 5: ANN v2 (8, Dropout 0.3)")
    m_v2 = run_ann_stage("ANN v2 (8)", (8,), X_train, y_train, X_test, y_test)

    # === Sonuçların toplanması ve yazdırılması ===
    results = {
        "Logistic Regression": m_base,
        "ANN v1 (16->8)": m_v1,
        "ANN v2 (8)": m_v2,
    }
    print_summary(results)

    # === Sonuçları diske kaydet ===
    # outputs/results.json — rapor ve sunum kaynaklı sayılar buradan
    # doğrulanabilir; jüri "bu rakamlar nereden?" diye sorarsa açacağımız yer.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
