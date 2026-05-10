from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/heart.csv")

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
PASSTHROUGH = ["sex", "fbs", "exang", "ca"]
TARGET = "condition"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(sparse_output=False), CATEGORICAL),
            ("pass", "passthrough", PASSTHROUGH),
        ]
    )


def get_splits(path: Path | str = DATA_PATH):
    df = load_raw(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, pre


def feature_names(pre: ColumnTransformer) -> list[str]:
    return pre.get_feature_names_out().tolist()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre = get_splits()
    names = feature_names(pre)
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"y_test  positive rate: {y_test.mean():.3f}")
    print(f"features ({len(names)}):")
    for n in names:
        print(f"  - {n}")
