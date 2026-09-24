"""
eda.py
------
Exploratory Data Analysis (EDA) on the real sentiment dataset.
No fake/invented values — everything printed here is computed from the actual CSV.

Run with:
    python eda.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

DATASET_PATH = os.path.join("dataset", "sentiment_dataset.csv")
CHARTS_DIR = "eda_charts"
REQUIRED_COLUMNS = {"text", "sentiment"}


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print("=" * 70)
        print("ERROR: Dataset file not found!")
        print(f"Expected location: {os.path.abspath(path)}")
        print("Please place your CSV file there with columns: text, sentiment")
        print("=" * 70)
        sys.exit(1)

    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"ERROR: Dataset is missing required column(s): {missing_cols}")
        sys.exit(1)
    return df


def main():
    print("\n=========== EXPLORATORY DATA ANALYSIS ===========\n")

    df = load_dataset(DATASET_PATH)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # --- Basic info ---
    print(f"[Shape] Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(f"[Columns] {list(df.columns)}")

    # --- Missing values ---
    print("\n[Missing Values per Column]")
    print(df.isnull().sum().to_string())

    # --- Duplicate rows ---
    duplicate_count = df.duplicated().sum()
    print(f"\n[Duplicate Rows] {duplicate_count}")

    # --- Sentiment distribution ---
    if "sentiment" in df.columns:
        print("\n[Sentiment Distribution]")
        sentiment_counts = df["sentiment"].astype(str).str.strip().str.lower().value_counts()
        print(sentiment_counts.to_string())

        # Chart: sentiment distribution (bar chart)
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336", "#9E9E9E"])
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        chart_path = os.path.join(CHARTS_DIR, "sentiment_distribution.png")
        plt.savefig(chart_path)
        plt.close(fig)
        print(f"[Saved] {chart_path}")

    # --- Text length statistics ---
    if "text" in df.columns:
        text_lengths = df["text"].astype(str).apply(len)
        print("\n[Text Length Statistics (characters)]")
        print(text_lengths.describe().to_string())

        # Chart: text length distribution (histogram)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(text_lengths, bins=20, color="#2196F3", edgecolor="black")
        ax.set_title("Text Length Distribution")
        ax.set_xlabel("Number of characters")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        chart_path = os.path.join(CHARTS_DIR, "text_length_distribution.png")
        plt.savefig(chart_path)
        plt.close(fig)
        print(f"[Saved] {chart_path}")

    print("\n=========== EDA COMPLETE ===========\n")


if __name__ == "__main__":
    main()
