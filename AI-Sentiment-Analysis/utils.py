"""
utils.py
--------
Reusable NLP text-preprocessing utilities for the sentiment analysis project.

Beginner note:
This file only PREPARES text (cleans it up) so a machine learning model can
understand it better. It does NOT do any prediction or training itself.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------------------------
# 1. Make sure the NLTK resources we need are available.
#    If they are missing, they will be downloaded automatically the first
#    time this file is imported (only happens once on a machine).
# ---------------------------------------------------------------------------
_REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _ensure_nltk_resources():
    """Download any missing NLTK resources quietly."""
    for name, path in _REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                # Some resources (e.g. punkt_tab) may not exist on older
                # NLTK versions. We ignore failures here and fall back to
                # a simple regex-based tokenizer inside preprocess_text().
                pass


_ensure_nltk_resources()

# ---------------------------------------------------------------------------
# 2. Build the lemmatizer and the stopword list once (reused for speed).
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback minimal stopword list in case download failed (e.g. no internet)
    _STOPWORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "and", "or", "but", "if", "of", "at", "by", "for", "with", "about",
        "to", "in", "on", "it", "this", "that", "i", "you", "he", "she",
        "we", "they", "them", "his", "her", "its", "our", "your", "their",
    }

# IMPORTANT: Negation words carry a lot of sentiment meaning
# (e.g. "not good" is very different from "good").
# We must NEVER remove these words, even though they are normally stopwords.
_NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "nor", "cannot", "can't", "cant", "won't", "wont", "don't", "dont",
    "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't",
    "arent", "wasn't", "wasnt", "weren't", "werent", "shouldn't",
    "shouldnt", "couldn't", "couldnt", "wouldn't", "wouldnt", "haven't",
    "havent", "hasn't", "hasnt", "hadn't", "hadnt",
}

# Keep negation words OUT of the stopword-removal set.
_STOPWORDS = _STOPWORDS - _NEGATION_WORDS

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_MULTISPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Clean and normalize a piece of text so it is ready for TF-IDF.

    Steps:
        1. Handle invalid/empty input safely.
        2. Lowercase the text.
        3. Remove URLs.
        4. Remove HTML tags.
        5. Remove punctuation/special characters (but keep negation words intact).
        6. Remove extra whitespace.
        7. Tokenize into words.
        8. Remove stopwords (except negation words like "not", "no", "never").
        9. Lemmatize each remaining word (e.g. "running" -> "run").

    Parameters
    ----------
    text : str
        Raw input text (e.g. a user review).

    Returns
    -------
    str
        Cleaned, space-joined text. Returns an empty string for invalid input.
    """
    # --- Step 1: Safety check for invalid/empty input ---
    if text is None or not isinstance(text, str):
        return ""
    text = text.strip()
    if text == "":
        return ""

    # --- Step 2: Lowercase ---
    text = text.lower()

    # --- Step 3: Remove URLs ---
    text = _URL_PATTERN.sub(" ", text)

    # --- Step 4: Remove HTML tags ---
    text = _HTML_PATTERN.sub(" ", text)

    # --- Step 5: Remove punctuation, but keep apostrophes in negation
    #             contractions (e.g. "don't") so they aren't broken apart. ---
    # Replace punctuation (except apostrophe) with a space.
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans(punctuation_to_remove, " " * len(punctuation_to_remove)))

    # --- Step 6: Collapse multiple spaces into one ---
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    if text == "":
        return ""

    # --- Step 7: Tokenize ---
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback if NLTK tokenizer data isn't available: simple split.
        tokens = text.split()

    # --- Step 8: Remove stopwords (negation words are protected above) ---
    tokens = [tok for tok in tokens if tok not in _STOPWORDS and tok.strip() != ""]

    # --- Step 9: Lemmatize ---
    lemmatized_tokens = [_lemmatizer.lemmatize(tok) for tok in tokens]

    return " ".join(lemmatized_tokens)


if __name__ == "__main__":
    # Quick manual test when running: python utils.py
    samples = [
        "I do NOT like this product at all! Visit http://example.com for details.",
        "<b>This is amazing!!!</b>   So many    extra spaces...",
        "",
        None,
    ]
    for s in samples:
        print(f"RAW:     {s!r}")
        print(f"CLEANED: {preprocess_text(s)!r}")
        print("-" * 50)
