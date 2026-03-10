import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📩 SMS Spam / Ham Classifier")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

@st.cache_resource
def train_model():

    df = pd.read_csv(
        r"C:\Users\KIRAN GHANTASALA\Downloads\spamhamdata.csv",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

    # Clean text
    df["message"] = df["message"].astype(str).apply(preprocess)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # -------- BAG OF WORDS --------
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, accuracy, cm


model, vectorizer, accuracy, cm = train_model()

st.subheader("📊 Model Performance")
st.write(f"Accuracy: **{accuracy:.4f}**")

st.subheader("📨 Enter Your SMS Message")
user_input = st.text_area("Type your message here")

if st.button("Predict"):

    if user_input.strip() != "":

        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 SPAM (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ HAM (Confidence: {probability:.2%})")

    else:
        st.warning("Please enter a message")