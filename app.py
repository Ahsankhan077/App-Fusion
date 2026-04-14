import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Predict
def predict_sentiment(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

# Page config
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="centered")

# Custom CSS (UI styling 🔥)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4A90E2;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🤖 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze text sentiment instantly</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_area("✍️ Enter your text here:", height=150)

# Button centered
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze = st.button("🔍 Analyze Sentiment")

# Prediction
if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        result = predict_sentiment(user_input)

        if result == 1:
            st.markdown(
                '<div class="result-box" style="background-color:#d4edda; color:#155724;">😊 Positive Sentiment</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box" style="background-color:#f8d7da; color:#721c24;">😡 Negative Sentiment</div>',
                unsafe_allow_html=True
            )
