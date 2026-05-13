from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from src.evaluate import evaluate
from src.preprocess import RANDOM_STATE, get_splits


EPOCHS = 200
BATCH_SIZE = 32
VAL_SIZE = 0.2
PATIENCE = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.3
DECISION_THRESHOLD = 0.5


def build_model(n_features: int, hidden_units=(16, 8), dropout: float = DROPOUT) -> Sequential:
    layers = [Input(shape=(n_features,))]
    for units in hidden_units:
        layers.append(Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(Dropout(dropout))
    layers.append(Dense(1, activation="sigmoid"))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model


def train_model(model, X_train, y_train, *, verbose=0):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_splits()

    architectures = {
        "ANN v1 (16->8, Dropout 0.3)": (16, 8),
        "ANN v2 (8, Dropout 0.3)":     (8,),
    }

    for name, hidden in architectures.items():
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        model = build_model(n_features=X_train.shape[1], hidden_units=hidden)
        history = train_model(model, X_train, y_train, verbose=0)
        epochs_run = len(history.history["loss"])
        print(f"\n[{name}] trained for {epochs_run} epochs (early stopping, patience={PATIENCE})")

        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        evaluate(name, y_test, y_pred, y_proba)
