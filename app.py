import streamlit as st
import pandas as pd
import joblib

# ------------------------
# Load saved model & scaler
# ------------------------
knn = joblib.load("crop_knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

# ------------------------
# Background Image
# ------------------------
page_bg_img = '''
<style>
body {
background-image: url("https://images.pexels.com/photos/6216870/pexels-photo-6216870.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260");
background-size: cover;
background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------
# Styled Title and Subtitle
# ------------------------
st.markdown(
    "<h1 style='color: white; text-align: center; font-size: 48px;'>🌾 Crop Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color: white; text-align: center; font-size: 20px;'>Predict the best crop based on soil nutrients & climate conditions</p>",
    unsafe_allow_html=True
)

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("Enter Soil & Climate Parameters")
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 140, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict Crop"):
    sample = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                          columns=['N','P','K','temperature','humidity','ph','rainfall'])
    sample_scaled = scaler.transform(sample)
    prediction = knn.predict(sample_scaled)
    
    # Display prediction in a colored box
    st.markdown(
        f"<div style='background-color:#4CAF50; padding:15px; border-radius:10px; text-align:center;'>"
        f"<h2 style='color:white;'>🌱 Recommended Crop: {prediction[0].capitalize()}</h2>"
        f"</div>", 
        unsafe_allow_html=True
    )

# ------------------------
# Optional: Show input parameters
# ------------------------
if st.checkbox("Show Input Parameters"):
    sample = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                          columns=['N','P','K','temperature','humidity','ph','rainfall'])
    st.table(sample)
