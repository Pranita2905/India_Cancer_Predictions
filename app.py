import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Cancer Survival Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load Model
with open("DecisionTree.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.title {
    text-align: center;
    color: #1e40af;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 30px;
}
.result {
    padding: 20px;
    border-radius: 10px;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🩺 Cancer Survival Prediction System</p>',
            unsafe_allow_html=True)

st.markdown('<p class="subtitle">Decision Tree Machine Learning Model</p>',
            unsafe_allow_html=True)

# Input Form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.number_input("Gender (Encoded)", value=0)
    state = st.number_input("State (Encoded)", value=0)
    city = st.number_input("City (Encoded)", value=0)

with col2:
    cancer_type = st.number_input("Cancer Type (Encoded)", value=0)
    stage = st.number_input("Stage (Encoded)", value=0)
    treatment_type = st.number_input("Treatment Type (Encoded)", value=0)
    survival_months = st.number_input("Survival Months", value=12)

# Prediction
if st.button("🔍 Predict Survival"):

    features = np.array([[
        age,
        gender,
        state,
        city,
        cancer_type,
        stage,
        treatment_type,
        survival_months
    ]])

    prediction = model.predict(features)[0]

    st.success(f"Prediction Result: {prediction}")

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed by Pranita | Data Analyst & Machine Learning Project</center>",
    unsafe_allow_html=True
)
