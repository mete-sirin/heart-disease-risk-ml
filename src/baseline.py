"""Logistic Regression baseline for ANN comparison."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate
from src.preprocess import RANDOM_STATE, get_splits


def train_baseline(X_train, y_train) -> LogisticRegression:
    # === Lojistik Regresyon — neden baseline olarak seçildi? ===
    # ANN'in gerçekten "ek değer" üretip üretmediğini ölçmek için bir
    # referansa ihtiyacımız var. Lojistik regresyon hızlı, yorumlanabilir
    # ve doğrusal sınırlar çizen klasik bir sınıflayıcı. Eğer veri doğrusal
    # ayrılabilirse zaten yüksek skor verir — ki bizim verimiz öyle.
    #
    # max_iter=1000: küçük ve dengeli verimizde varsayılan 100 iter zaman
    # zaman convergence uyarısı veriyor; 1000'e çıkarıp güvenceye aldık.
    # Hyperparameter taraması YAPILMADI; bu özellikle bilinçli bir karar:
    # baseline'ın tuning'siz değerini görmek istedik.
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


# === Modülü doğrudan çalıştırmak için ===
# `python -m src.baseline` ile yalnız baseline'ı çalıştırıp metriklerini
# görebilirsiniz. Sunum sırasında ANN bölümünden önce hızlı bir
# karşılaştırma yapmak isterseniz pratik.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_splits()
    model = train_baseline(X_train, y_train)

    # Hem sınıf tahmini (predict) hem olasılık tahmini (predict_proba)
    # alıyoruz. ROC-AUC olasılık skoruna ihtiyaç duyar.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    evaluate("Logistic Regression (baseline)", y_test, y_pred, y_proba)
