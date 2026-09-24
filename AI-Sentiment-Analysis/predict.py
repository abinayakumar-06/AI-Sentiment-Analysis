"""
predict.py
----------
Loads the trained model + TF-IDF vectorizer and predicts sentiment for new text.

This module does NOT retrain anything — it only loads what train_model.py
already saved to the 'model/' folder.
"""

import os

import joblib

from utils import preprocess_text

MODEL_PATH = os.path.join("model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")

# Cache loaded objects so we don't reload from disk on every single call.
_model = None
_vectorizer = None


class ModelNotFoundError(Exception):
    """Raised when the trained model or vectorizer files cannot be found."""
    pass


def _load_artifacts():
    """Load the model and vectorizer from disk (only once, then cached)."""
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please run 'python train_model.py' first to train and save the model."
        )
    if not os.path.exists(VECTORIZER_PATH):
        raise ModelNotFoundError(
            f"Vectorizer file not found at '{VECTORIZER_PATH}'. "
            "Please run 'python train_model.py' first to train and save the vectorizer."
        )

    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def predict_sentiment(text: str) -> dict:
    """
    Predict the sentiment of a piece of text.

    Parameters
    ----------
    text : str
        Raw input text (e.g. a review).

    Returns
    -------
    dict
        {
            "sentiment": "positive" | "negative" | "neutral" | None,
            "confidence": float between 0 and 1, or None if unavailable,
            "error": str or None
        }
    """
    result = {"sentiment": None, "confidence": None, "error": None}

    # --- Validate input ---
    if text is None or not isinstance(text, str) or text.strip() == "":
        result["error"] = "Input text is empty or invalid. Please enter some text."
        return result

    # --- Load model/vectorizer (raises a friendly error if missing) ---
    try:
        model, vectorizer = _load_artifacts()
    except ModelNotFoundError as e:
        result["error"] = str(e)
        return result

    # --- Preprocess the input the same way training data was processed ---
    cleaned = preprocess_text(text)
    if cleaned == "":
        result["error"] = "Input text became empty after cleaning (e.g. only stopwords/punctuation)."
        return result

    # --- Transform with the SAME fitted vectorizer used during training ---
    try:
        features = vectorizer.transform([cleaned])
    except Exception as e:
        result["error"] = f"Failed to vectorize input text: {e}"
        return result

    # --- Predict ---
    try:
        prediction = model.predict(features)[0]
        result["sentiment"] = prediction

        # Confidence is only available if the model supports predict_proba()
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            class_index = list(model.classes_).index(prediction)
            result["confidence"] = float(probabilities[class_index])
    except Exception as e:
        result["error"] = f"Prediction failed: {e}"

    return result


if __name__ == "__main__":
    # Quick manual test when running: python predict.py
    test_texts = [
        "I absolutely love this product, it works great!",
        "This is the worst service I have ever experienced.",
        "The package arrived on Tuesday.",
    ]
    for t in test_texts:
        outcome = predict_sentiment(t)
        print(f"Text: {t}")
        print(f"Result: {outcome}")
        print("-" * 50)
