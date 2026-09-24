"""
train_model.py
---------------
Trains a Logistic Regression sentiment classifier using TF-IDF features.

Pipeline:
    Load Dataset -> Clean Dataset -> Preprocess Text -> Train/Test Split ->
    TF-IDF Vectorization -> Logistic Regression -> Prediction -> Evaluation -> Save Model

Run with:
    python train_model.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import preprocess_text

# ---------------------------------------------------------------------------
# Paths (relative, so this works on any machine / OS, including Windows)
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "sentiment_dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

REQUIRED_COLUMNS = {"text", "sentiment"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and validate it exists and has the right columns."""
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
        print("=" * 70)
        print(f"ERROR: Dataset is missing required column(s): {missing_cols}")
        print(f"Found columns: {list(df.columns)}")
        print("Expected columns: 'text' and 'sentiment'")
        print("=" * 70)
        sys.exit(1)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, empty text, and invalid sentiment labels."""
    original_len = len(df)

    # Drop rows where text or sentiment is missing
    df = df.dropna(subset=["text", "sentiment"])

    # Normalize sentiment labels (lowercase, stripped)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Remove rows with invalid sentiment labels
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)]

    # Remove empty / whitespace-only text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Remove duplicate rows
    duplicates_found = df.duplicated(subset=["text", "sentiment"]).sum()
    df = df.drop_duplicates(subset=["text", "sentiment"])

    cleaned_len = len(df)
    print(f"[Data Cleaning] Original rows: {original_len}")
    print(f"[Data Cleaning] Duplicate rows removed: {duplicates_found}")
    print(f"[Data Cleaning] Rows after cleaning (missing/invalid/empty removed): {cleaned_len}")

    if cleaned_len == 0:
        print("ERROR: No valid rows remain after cleaning. Please check your dataset.")
        sys.exit(1)

    return df.reset_index(drop=True)


def check_class_balance(df: pd.DataFrame) -> None:
    """Print sentiment class distribution and warn about imbalance."""
    counts = df["sentiment"].value_counts()
    print("\n[Class Distribution]")
    print(counts.to_string())

    max_count = counts.max()
    min_count = counts.min()
    if min_count > 0 and (max_count / min_count) >= 1.5:
        print(
            "\n[WARNING] The dataset appears imbalanced "
            f"(largest class is {max_count / min_count:.1f}x the smallest). "
            "Using 'class_weight=balanced' in Logistic Regression to help compensate."
        )
    else:
        print("\n[Info] Class distribution looks reasonably balanced.")


def main():
    print("\n=========== SENTIMENT MODEL TRAINING PIPELINE ===========\n")

    # --- 1. Load dataset ---
    df = load_dataset(DATASET_PATH)
    print(f"[Load] Loaded {len(df)} rows from {DATASET_PATH}")

    # --- 2. Clean dataset ---
    df = clean_dataset(df)

    # --- Check class balance (used later to decide on class_weight) ---
    check_class_balance(df)

    # --- 3. Preprocess text ---
    print("\n[Preprocessing] Cleaning and normalizing text (this may take a moment)...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Remove rows that became empty after preprocessing (e.g. text was only stopwords)
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    removed = before - len(df)
    if removed > 0:
        print(f"[Preprocessing] Removed {removed} rows that became empty after cleaning.")

    if len(df) < 10:
        print("ERROR: Not enough data remains to train a reliable model (need at least ~10 rows).")
        sys.exit(1)

    # --- 4. Train/test split (stratified to keep class proportions) ---
    X = df["clean_text"]
    y = df["sentiment"]

    test_size = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # Stratify can fail if a class has very few samples; fall back gracefully.
        print("[Warning] Could not stratify split (a class may have too few samples). Using a plain split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    print(f"\n[Split] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # --- 5. TF-IDF Vectorization ---
    # Fit ONLY on training data to avoid data leakage into the test set.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams + bigrams (helps capture phrases like "not good")
        min_df=1,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)  # transform only, never fit on test data

    # --- 6. Train Logistic Regression model ---
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # helps if classes are imbalanced
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    # --- 7. Predict on the test set ---
    y_pred = model.predict(X_test_tfidf)

    # --- 8. Evaluation (real, computed metrics — never hardcoded) ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n=========== MODEL EVALUATION RESULTS ===========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- Confusion matrix ---
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("[Confusion Matrix] (rows = actual, columns = predicted)")
    print(f"Labels order: {labels}")
    print(cm)

    # Save confusion matrix as an image for easy viewing
    os.makedirs("eda_charts", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join("eda_charts", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"\n[Saved] Confusion matrix image -> {cm_path}")

    # --- 9. Save model and vectorizer using Joblib ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\n[Saved] Model      -> {MODEL_PATH}")
    print(f"[Saved] Vectorizer -> {VECTORIZER_PATH}")
    print("\n=========== TRAINING COMPLETE ===========\n")


if __name__ == "__main__":
    main()
