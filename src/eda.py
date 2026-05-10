from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import CATEGORICAL, NUMERIC, PASSTHROUGH, TARGET, load_raw


OUTPUT_DIR = Path("outputs/eda")

sns.set_theme(style="whitegrid", palette="Set2")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_balance(df: pd.DataFrame) -> Path:
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
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, NUMERIC):
        for cls, sub in df.groupby(TARGET):
            sns.kdeplot(sub[feat], ax=ax, label=f"condition={cls}",
                        fill=True, alpha=0.35, linewidth=1.5)
        ax.set_title(feat)
        ax.legend()
    for j in range(len(NUMERIC), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sayısal özelliklerin sınıfa göre dağılımı", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "02_numerical_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_categorical_distributions(df: pd.DataFrame) -> Path:
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


if __name__ == "__main__":
    run()
