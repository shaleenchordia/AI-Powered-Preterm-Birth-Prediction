import streamlit as st
import requests
import json
import joblib
import numpy as np
import ssl
import os
import urllib3
from dotenv import load_dotenv

# 🔹 Fix SSL Issues (For API Requests)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🔹 Load environment variables
load_dotenv()
API_KEY = os.getenv("GROK_API_KEY")

if not API_KEY:
    st.error("❌ API key missing! Make sure it's set in .env")
    st.stop()

# 🔹 Grok API Details
GROK_API_URL = "https://api.grok.ai/v1/chat/completions"  # Replace with actual endpoint

# 🔹 Function to call Grok API
def get_llm_response(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "grok-1", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=15, verify=False)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.SSLError as e:
        return f"🔴 SSL Error: {e}"
    except requests.exceptions.Timeout:
        return "⚠️ API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {e}"

# 🔹 Load Trained Model
MODEL_PATH = "mlp_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict"):
        raise ValueError("Invalid model format: 'predict' method missing!")
except FileNotFoundError:
    st.error("🚨 Model file not found! Ensure 'mlp_model.pkl' is in the same directory.")
    st.stop()
except ValueError as e:
    st.error(f"🚨 Model Load Error: {e}")
    st.stop()

# 🔹 Streamlit UI
st.title("Preterm Birth Prediction & LLM Integration")

st.header("Enter Pregnancy Data for Prediction")
age_group = st.number_input("Age Group (e.g., 1-5):", min_value=1, max_value=5, step=1)
reported_race_ethnicity = st.number_input("Race/Ethnicity Code:", min_value=0, max_value=10, step=1)
previous_births = st.number_input("Number of Previous Births:", min_value=0.0, step=1.0)
tobacco_use = st.radio("Tobacco Use During Pregnancy:", [0, 1])
prenatal_care = st.radio("Received Adequate Prenatal Care:", [0, 1])

if st.button("Predict Preterm Birth Risk"):
    user_input = np.array([[age_group, reported_race_ethnicity, previous_births, tobacco_use, prenatal_care]])
    prediction = model.predict(user_input)[0]
    st.success(f"Preterm Birth Prediction: {'Yes' if prediction == 1 else 'No'}")

# 🔹 LLM Section
st.header("Ask LLM for More Insights")
user_query = st.text_area("Ask anything about preterm birth:")

if st.button("Get LLM Response"):
    llm_response = get_llm_response(user_query)
    st.write("🧠 Grok's Response:", llm_response)
