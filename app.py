import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Sentiment Analysis App 😊")

text = st.text_input("Enter your text")

if st.button("Predict"):

    # 🔹 Check if input is empty
    if text.strip() == "":
        st.warning("Please enter some text!")

    # 🔹 Check if input has NO alphabets (only numbers/special chars)
    elif not re.search("[a-zA-Z]", text):
        st.error("Please enter valid text (not only numbers or symbols) ❌")

    else:
        data = vectorizer.transform([text])
        result = model.predict(data)[0]

        if result == "positive":
            st.success("Positive 😊")
        elif result == "negative":
            st.error("Negative 😞")
        else:
            st.info("Neutral 😐")