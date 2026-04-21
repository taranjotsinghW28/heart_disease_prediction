import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter your health details below to check your 10-year risk")

# Load model and scaler
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Info")
    age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)
    systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120, step=1)
    hypertension = st.selectbox("Hypertension?", ["No", "Yes"])

with col2:
    st.subheader("Health Metrics")
    diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80, step=1)
    glucose = st.number_input("Glucose Level", min_value=70, max_value=300, value=120, step=1)
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

# Convert Yes/No to 1/0
hypertension_val = 1 if hypertension == "Yes" else 0
diabetes_val = 1 if diabetes == "Yes" else 0

# Predict button
if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    # Create input array (order must match training: age, sysBP, prevalentHyp, diaBP, glucose, diabetes)
    input_data = np.array([[age, systolic_bp, hypertension_val, diastolic_bp, glucose, diabetes_val]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Show results
    st.divider()
    
    if prediction == 1:
        st.error("⚠️ HIGH RISK of Heart Disease")
        st.metric("Risk Probability", f"{probability*100:.1f}%")
        st.warning("Please consult a doctor for preventive care")
    else:
        st.success("✅ LOW RISK of Heart Disease")
        st.metric("Risk Probability", f"{probability*100:.1f}%")
        st.info("Maintain a healthy lifestyle")
    
    # Additional info
    with st.expander("About this prediction"):
        st.write("""
        This prediction is based on the Random Forest model trained on Framingham Heart Study data.
        
        **Key risk factors considered:**
        - Age
        - Blood Pressure (Systolic & Diastolic)
        - Hypertension status
        - Glucose level
        - Diabetes status
        
        ⚠️ This is for educational purposes only. Not a substitute for medical advice.
        """)

# Footer
st.divider()
st.caption("Made with ❤️ for educational purposes | Data Source: Framingham Heart Study")