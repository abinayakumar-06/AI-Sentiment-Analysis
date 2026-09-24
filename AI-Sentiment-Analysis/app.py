"""
app.py
------
Streamlit web application for AI-Based Sentiment Analysis.

Run with:
    streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from predict import ModelNotFoundError, predict_sentiment

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
)

# Emoji + color lookup for each sentiment label
SENTIMENT_DISPLAY = {
    "positive": {"emoji": "😊", "label": "Positive", "color": "#4CAF50"},
    "negative": {"emoji": "😞", "label": "Negative", "color": "#F44336"},
    "neutral": {"emoji": "😐", "label": "Neutral", "color": "#9E9E9E"},
}

EXAMPLES = {
    "Positive": "The product is excellent and I really enjoyed using it.",
    "Negative": "The service was terrible and I am very disappointed.",
    "Neutral": "The product was delivered yesterday.",
}

# ---------------------------------------------------------------------------
# Session state initialization (used for the dashboard — real session stats only)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    # Each entry: {"text": str, "sentiment": str, "confidence": float or None}
    st.session_state.history = []

if "text_input" not in st.session_state:
    st.session_state.text_input = ""


def render_sentiment_result(result: dict):
    """Display a prediction result nicely, or a friendly error message."""
    if result.get("error"):
        st.error(f"⚠️ {result['error']}")
        return

    sentiment = result["sentiment"]
    confidence = result["confidence"]
    display = SENTIMENT_DISPLAY.get(sentiment, {"emoji": "❓", "label": sentiment, "color": "#000000"})

    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {display['color']}22;
                    border: 2px solid {display['color']}; text-align: center;">
            <h2 style="color: {display['color']}; margin: 0;">
                {display['emoji']} {display['label']}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.write("")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"Confidence: {confidence * 100:.1f}%")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This application uses Natural Language Processing, "
        "TF-IDF feature extraction, and Logistic Regression "
        "to classify text into Positive, Negative, or Neutral sentiment."
    )

    st.header("🛠️ Technologies")
    st.markdown(
        "- Python\n"
        "- NLP\n"
        "- NLTK\n"
        "- TF-IDF\n"
        "- Scikit-learn\n"
        "- Logistic Regression\n"
        "- Streamlit"
    )

    st.header("📊 Session Dashboard")
    total = len(st.session_state.history)
    pos = sum(1 for h in st.session_state.history if h["sentiment"] == "positive")
    neg = sum(1 for h in st.session_state.history if h["sentiment"] == "negative")
    neu = sum(1 for h in st.session_state.history if h["sentiment"] == "neutral")

    col1, col2 = st.columns(2)
    col1.metric("Total Predictions", total)
    col2.metric("😊 Positive", pos)
    col1.metric("😞 Negative", neg)
    col2.metric("😐 Neutral", neu)

    if total > 0:
        dist_df = pd.DataFrame(
            {"Sentiment": ["Positive", "Negative", "Neutral"], "Count": [pos, neg, neu]}
        ).set_index("Sentiment")
        st.bar_chart(dist_df)
    else:
        st.caption("Make a prediction to see the session dashboard fill up.")

    if total > 0 and st.button("🗑️ Clear Session Stats"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("🤖 AI-Based Sentiment Analysis")
st.subheader("Analyze the sentiment of your text using NLP and Machine Learning.")
st.divider()

# ---------------------------------------------------------------------------
# Check model files exist before letting the user interact.
# ---------------------------------------------------------------------------
model_exists = os.path.exists(os.path.join("model", "sentiment_model.pkl"))
vectorizer_exists = os.path.exists(os.path.join("model", "tfidf_vectorizer.pkl"))

if not (model_exists and vectorizer_exists):
    st.error(
        "⚠️ Trained model files were not found in the 'model/' folder.\n\n"
        "Please run the following command first, then restart this app:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Single Text Analysis
# ---------------------------------------------------------------------------
st.header("📝 Analyze a Single Text")

st.write("Try an example, or type your own text below:")
ex_col1, ex_col2, ex_col3 = st.columns(3)
if ex_col1.button("😊 Positive Example"):
    st.session_state.text_input = EXAMPLES["Positive"]
if ex_col2.button("😞 Negative Example"):
    st.session_state.text_input = EXAMPLES["Negative"]
if ex_col3.button("😐 Neutral Example"):
    st.session_state.text_input = EXAMPLES["Neutral"]

user_text = st.text_area(
    "Enter your review or text here...",
    value=st.session_state.text_input,
    height=150,
    key="text_area_input",
)

if st.button("🔍 Analyze Sentiment", type="primary"):
    if not user_text or user_text.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            outcome = predict_sentiment(user_text)
            render_sentiment_result(outcome)

            # Only log to session dashboard if prediction succeeded
            if outcome.get("sentiment") is not None:
                st.session_state.history.append(
                    {
                        "text": user_text,
                        "sentiment": outcome["sentiment"],
                        "confidence": outcome["confidence"],
                    }
                )
        except ModelNotFoundError as e:
            st.error(f"⚠️ {e}")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Batch CSV Analysis
# ---------------------------------------------------------------------------
st.header("📂 Batch Sentiment Analysis")
st.write("Upload a CSV file with a `text` column to analyze many rows at once.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Could not read the uploaded file as CSV: {e}")
        batch_df = None

    if batch_df is not None:
        if batch_df.empty:
            st.warning("The uploaded CSV file is empty.")
        elif "text" not in batch_df.columns:
            st.error("⚠️ The uploaded CSV must contain a column named 'text'.")
        else:
            # Handle missing text values
            missing_count = batch_df["text"].isnull().sum()
            valid_df = batch_df.dropna(subset=["text"]).copy()
            valid_df["text"] = valid_df["text"].astype(str)
            valid_df = valid_df[valid_df["text"].str.strip() != ""]

            if missing_count > 0:
                st.info(f"ℹ️ Skipped {missing_count} row(s) with missing text values.")

            if valid_df.empty:
                st.warning("No valid text rows found to analyze after removing missing/empty values.")
            else:
                with st.spinner(f"Analyzing {len(valid_df)} rows..."):
                    predicted_sentiments = []
                    confidences = []
                    for text_value in valid_df["text"]:
                        outcome = predict_sentiment(text_value)
                        predicted_sentiments.append(outcome.get("sentiment"))
                        confidences.append(outcome.get("confidence"))

                    valid_df["predicted_sentiment"] = predicted_sentiments
                    valid_df["confidence"] = confidences

                st.success(f"✅ Analyzed {len(valid_df)} rows successfully.")
                st.dataframe(valid_df, use_container_width=True)

                # Quick summary chart for the batch
                batch_counts = valid_df["predicted_sentiment"].value_counts()
                st.bar_chart(batch_counts)

                # Download button for the analyzed CSV
                csv_bytes = valid_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Analyzed CSV",
                    data=csv_bytes,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )

st.divider()
st.caption("Built with Python, NLP, TF-IDF, Logistic Regression, and Streamlit.")
