import streamlit as st
import joblib

# Load saved models
tfidf = joblib.load("tfidf.pkl")
classifier = joblib.load("classifier.pkl")
regressor = joblib.load("regressor.pkl")

# Page title
st.title("AutoJudge 🧠")
st.subheader("Predict Programming Problem Difficulty")

st.write(
    "Paste the problem description below and click **Predict** "
    "to get the difficulty class and score."
)

# Text input
problem_text = st.text_area(
    "Problem Description",
    height=300,
    placeholder="Paste problem title, description, input/output here..."
)

# Predict button
if st.button("Predict"):
    if problem_text.strip() == "":
        st.warning("Please enter a problem description.")
    else:
      
        X = tfidf.transform([problem_text])
        predicted_class = classifier.predict(X)[0]
        predicted_score = regressor.predict(X)[0]

      
        st.success("Prediction Complete ✅")
        st.write(f"### 📘 Difficulty Class: **{predicted_class}**")
        st.write(f"### 🔢 Difficulty Score: **{predicted_score:.2f}**")
