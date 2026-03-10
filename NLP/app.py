import streamlit as st
import pickle

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📝 Sentiment Analysis App")
st.write("Enter a sentence to predict whether it is Positive or Negative.")

# ---------------------------------------------------
# Load Pickle Files
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, tfidf = load_model()

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
user_input = st.text_area("Enter your sentence here:")

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a sentence.")
    else:
        transformed_input = tfidf.transform([user_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")