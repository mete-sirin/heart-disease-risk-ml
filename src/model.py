"""ANN: parameterized Dense+Dropout stack -> Dense(1, Sigmoid)."""

from __future__ import annotations

import os

# === TensorFlow log seviyesini düşür ===
# Varsayılan olarak TF, oneDNN bilgilendirmeleri ve CPU instruction uyarıları
# basıyor. Sunum sırasında konsolu temiz tutmak için INFO/WARNING'leri
# bastırıyoruz; ERROR seviyesi yine görünür.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from src.evaluate import evaluate
from src.preprocess import RANDOM_STATE, get_splits


# === Eğitim hyperparameter'ları ===
# Tüm değerler context.md spec'ine uygun. Sabitleri burada toplayarak
# herhangi bir denemede tek noktadan değiştirilebilir hale getirdik.
EPOCHS = 200            # Üst sınır; EarlyStopping çok daha erken durdurur.
BATCH_SIZE = 32         # Küçük veri setinde 32 makul; daha büyük batch noisy.
VAL_SIZE = 0.2          # Train'in %20'si validation, geri kalan model eğitimi.
PATIENCE = 20           # 20 epoch boyunca val_loss düşmezse eğitimi kes.
LEARNING_RATE = 1e-3    # Adam için varsayılan, küçük problemler için stabil.
DROPOUT = 0.3           # Spec'ten geliyor; aşırı öğrenmeyi engellemek için.
DECISION_THRESHOLD = 0.5  # Sigmoid çıktıyı sınıfa çevirme eşiği.


def build_model(n_features: int, hidden_units=(16, 8), dropout: float = DROPOUT) -> Sequential:
    # === Parametrik mimari kurucusu ===
    # hidden_units bir tuple; içinde kaç eleman varsa o kadar gizli katman
    # oluşur. Bu sayede aynı fonksiyondan hem v1 (16, 8) hem v2 (8,)
    # mimarisini kurabiliyoruz — kod tekrarı yok.
    layers = [Input(shape=(n_features,))]
    for units in hidden_units:
        layers.append(Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(Dropout(dropout))
    # Son katman: tek nöron + sigmoid → 0/1 olasılığı.
    layers.append(Dense(1, activation="sigmoid"))

    # === Derleme: optimizer, loss ve izlenecek metrikler ===
    # Loss = binary_crossentropy (ikili sınıflandırmanın standardı).
    # Metrik olarak accuracy + recall izleniyor. Recall'ı validation
    # tarafında da takip edebilmek için Keras metric'i ekledik.
    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model


def train_model(model, X_train, y_train, *, verbose=0):
    # === Validation split — neden Keras'ın default'unu kullanmadık? ===
    # Keras'ın validation_split parametresi train verisinin SON %20'sini
    # alır ve bunu STRATIFIYE ETMEZ. Küçük veri setimizde bu sınıf oranını
    # bozabilir. O yüzden train_test_split'i elle çağırıp stratify=y_train
    # ile sınıf dağılımını koruyarak validation setini ayırıyoruz.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # === Early stopping — overfitting'e karşı koruma ===
    # Validation loss 20 epoch boyunca iyileşmezse eğitim kesilir.
    # restore_best_weights=True: durulduğunda en iyi val_loss'u veren
    # ağırlıklara geri döner — yani son ağırlıkları değil, en iyisini alırız.
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


# === İki mimariyi karşılaştırmak için __main__ bloğu ===
# `python -m src.model` ile çalıştırıldığında v1 ve v2'yi sırayla eğitir
# ve metriklerini yazdırır. Sunumdaki §6 ve §7 sonuçlarını üreten yer.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_splits()

    # Karşılaştırılacak iki mimari:
    # v1: spec mimarisi (iki gizli katman) — baseline'ın altında kaldı
    # v2: tek katman, daha az kapasite — kazanan mimari
    architectures = {
        "ANN v1 (16->8, Dropout 0.3)": (16, 8),
        "ANN v2 (8, Dropout 0.3)":     (8,),
    }

    for name, hidden in architectures.items():
        # Her model build'inden ÖNCE seed'i resetliyoruz; iki çalıştırma
        # birbirini etkilemesin diye. Reproducibility için kritik.
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        model = build_model(n_features=X_train.shape[1], hidden_units=hidden)
        history = train_model(model, X_train, y_train, verbose=0)
        epochs_run = len(history.history["loss"])
        print(f"\n[{name}] trained for {epochs_run} epochs (early stopping, patience={PATIENCE})")

        # Olasılık tahminini sınıfa dönüştürmek için 0.5 eşiği kullanılıyor.
        # Bu eşiği aşağı çekerek recall'ı yükseltmek mümkün; gelecek iş.
        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        evaluate(name, y_test, y_pred, y_proba)
