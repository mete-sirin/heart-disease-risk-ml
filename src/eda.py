"""Exploratory data analysis: distributions, correlations, target balance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import CATEGORICAL, NUMERIC, PASSTHROUGH, TARGET, load_raw


# === Çıktı klasörü ve grafik teması ===
# Tüm figürler outputs/eda/ altına kaydedilir; bu klasör .gitignore'da
# whitelist edildi, yani figürler raporla birlikte versiyonlanır.
# Set2 paleti pastel ama yeterince ayrıştırıcı renkler verir.
OUTPUT_DIR = Path("outputs/eda")

sns.set_theme(style="whitegrid", palette="Set2")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_balance(df: pd.DataFrame) -> Path:
    # === Hedef sınıf dağılımı (Slayt 2'de kullanılan figür) ===
    # 160 negatif vs 137 pozitif — hafif dengesizlik var ama ciddi sorun değil.
    # Bu yüzden class_weight veya SMOTE gibi tekniklere ihtiyaç duymadık.
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                hue=counts.index.astype(str), legend=False)
    ax.set_title("Hedef sınıf dağılımı (condition)")
    ax.set_xlabel("condition")
    ax.set_ylabel("Örnek sayısı")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    path = OUTPUT_DIR / "01_target_balance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_numerical_distributions(df: pd.DataFrame) -> Path:
    # === Sayısal özelliklerin sınıfa göre KDE'si ===
    # Her sayısal özellik için iki sınıfın yoğunluk eğrisini üst üste çiziyoruz.
    # Sunumda en çok dikkat çeken: thalach (hastalarda max kalp atışı belirgin
    # şekilde düşük) ve oldpeak (pozitif sınıfın sağ kuyruğu uzun).
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, NUMERIC):
        for cls, sub in df.groupby(TARGET):
            sns.kdeplot(sub[feat], ax=ax, label=f"condition={cls}",
                        fill=True, alpha=0.35, linewidth=1.5)
        ax.set_title(feat)
        ax.legend()
    # 5 sayısal özellik var, 6 alt grafiğin sonuncusu boş kalır
    for j in range(len(NUMERIC), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sayısal özelliklerin sınıfa göre dağılımı", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "02_numerical_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_categorical_distributions(df: pd.DataFrame) -> Path:
    # === Kategorik ve ikili özelliklerin sınıfa göre count plot'u ===
    # 8 alt grafik (4 kategorik + 4 ikili/tamsayı). En güçlü ayırıcılar:
    # thal=2 ve ca>=1 pozitif sınıfa, thal=0 ve ca=0 negatif sınıfa yığılıyor.
    # fbs neredeyse hiç ayrım göstermiyor — bilgi taşımayan sütun.
    feats = CATEGORICAL + PASSTHROUGH
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, feats):
        sns.countplot(data=df, x=feat, hue=TARGET, ax=ax, palette="Set2")
        ax.set_title(feat)
        ax.set_xlabel("")
    fig.suptitle("Kategorik ve ikili özelliklerin sınıfa göre dağılımı", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "03_categorical_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    # === Pearson korelasyon ısı haritası ===
    # Bu grafik sunumun en önemli görsellerinden biri: hedefle en yüksek
    # korelasyon thal'da (0.52). Ayrıca özellikler arası korelasyonların
    # zayıf olması (|r|<0.5) çoklu eş-doğrusallık olmadığını gösteriyor.
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson korelasyon matrisi", fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / "04_correlation_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def print_summary(df: pd.DataFrame) -> None:
    # === Konsol özeti — figürlere ek olarak metin tabanlı rapor ===
    # Şekil, eksik değer durumu, hedef dağılımı, sayısal istatistikler ve
    # hedefle en yüksek korelasyona sahip 8 özelliğin listesini yazdırır.
    # Sunum sırasında main.py çıktısında bu blok ilk akan kısımdır.
    print("=== Şekil ===")
    print(df.shape)

    print("\n=== Eksik değerler ===")
    miss = df.isnull().sum()
    print(f"toplam: {int(miss.sum())}")

    print("\n=== Hedef dağılımı ===")
    counts = df[TARGET].value_counts().sort_index()
    print(counts.to_string())
    print(f"pozitif oranı: {df[TARGET].mean():.3f}")

    print("\n=== Sayısal özellikler özeti ===")
    print(df[NUMERIC].describe().round(2).to_string())

    print("\n=== Hedefle |korelasyon|, en yüksek 8 ===")
    target_corr = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(target_corr.head(8).round(3).to_string())


def run() -> list[Path]:
    # === EDA pipeline'ının orkestrasyonu ===
    # main.py "STAGE 1: Exploratory Data Analysis" aşamasında bu fonksiyonu
    # çağırır. Önce konsol özeti, sonra 4 figür kaydedilir.
    ensure_output_dir()
    df = load_raw()
    print_summary(df)
    paths = [
        plot_target_balance(df),
        plot_numerical_distributions(df),
        plot_categorical_distributions(df),
        plot_correlation_heatmap(df),
    ]
    print("\n=== Kaydedilen grafikler ===")
    for p in paths:
        print(f"  - {p}")
    return paths


# === Bağımsız çalıştırma desteği ===
# `python -m src.eda` ile sadece EDA aşamasını ayrı çalıştırabilirsiniz.
if __name__ == "__main__":
    run()
