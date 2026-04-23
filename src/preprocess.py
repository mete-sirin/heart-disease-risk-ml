"""Load data/heart.csv, one-hot encode categoricals, scale numerics, stratified split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# === Veri dosyası yolu ===
# Tek bir kaynak: data/heart.csv. Tüm pipeline bu yoldan okuyor.
DATA_PATH = Path("data/heart.csv")


# === Sütun grupları ===
# Üç farklı işlem grubu: sayısallar standartlaştırılır, kategorikler one-hot
# olur, ikili/tamsayı sütunlar olduğu gibi geçer. Bu bölümleme context.md'deki
# "Features" tablosuyla birebir aynıdır — tek doğru kaynak burada tutuluyor.
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
PASSTHROUGH = ["sex", "fbs", "exang", "ca"]
TARGET = "condition"


# === Reproducibility sabitleri ===
# random_state=42: aynı kodu tekrar çalıştıran herkes aynı split'i alır.
# Sunumda jüri "kodu çalıştırınca aynı sonucu alır mıyız?" derse cevap evet.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    # Ham CSV'yi olduğu gibi döner — EDA bunu doğrudan kullanır.
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    # === Tek bir ColumnTransformer içinde üç dönüşüm ===
    # ColumnTransformer kullanmamızın amacı: aynı pipeline'ı hem train'de fit
    # ederken hem test'te transform ederken **tutarlı** uygulamak. Manuel
    # yapsaydık bir sütunu unutmak veya farklı sıraya koymak çok kolaydı.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(sparse_output=False), CATEGORICAL),
            ("pass", "passthrough", PASSTHROUGH),
        ]
    )


def get_splits(path: Path | str = DATA_PATH):
    # === Veri setini yükle ve hedef sütunu ayır ===
    df = load_raw(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy()

    # === Stratified train/test split ===
    # stratify=y ile sınıf oranı her iki sette de korunuyor. Küçük veri
    # setlerinde bu önemli — rastgele bir split, test setine bütün pozitifleri
    # ya da bütün negatifleri yığabilir.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # === Data leakage'a karşı kritik adım ===
    # Scaler ve OneHotEncoder YALNIZCA train'de fit ediliyor (.fit_transform).
    # Test setine sadece transform uyguluyoruz. Aksi halde test setinin
    # ortalama/std değerleri eğitime sızar ve metrikler iyimser çıkar.
    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, pre


def feature_names(pre: ColumnTransformer) -> list[str]:
    # ColumnTransformer dönüşüm sonrası 22 sütun üretir; bunların adlarını
    # alıp interpretasyon/raporlama için kullanılabilir hale getiriyoruz.
    return pre.get_feature_names_out().tolist()


# === Modülü doğrudan çalıştırmak için sanity-check bloğu ===
# `python -m src.preprocess` ile çalıştırıldığında split şekillerini ve
# özellik isimlerini yazdırır. Pipeline'ın doğru kurulduğunu görmek için.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre = get_splits()
    names = feature_names(pre)
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"y_test  positive rate: {y_test.mean():.3f}")
    print(f"features ({len(names)}):")
    for n in names:
        print(f"  - {n}")
