import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon=":speech_balloon:",
    layout="centered"
)

st.title("Twitter Sentiment Analysis")
st.write("Enter a text message to analyze its sentiment using BERT.")

text = st.text_area(
    "Enter text",
    placeholder="Example: I love this product!",
    height=150
)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=30
            )

            if response.ok:
                result = response.json()

                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.success(f"Sentiment: {sentiment}")
                st.info(f"Confidence: {confidence:.2%}")
            else:
                st.error(f"API error: {response.text}")

        except requests.RequestException as e:
            st.error(f"Could not connect to API: {e}")
