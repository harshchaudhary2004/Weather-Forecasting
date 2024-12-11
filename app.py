import streamlit as st
import joblib
import numpy as np

# Load the model and label encoder
model = joblib.load('weather_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Weather Prediction App 🌦️")
st.markdown("**Enter weather parameters below to predict the weather condition.**")

# Input fields
temperature = st.number_input("Temperature (Celsius):", min_value=-50.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Wind Speed (km/h):", min_value=0.0, max_value=200.0, value=10.0)

# Predict button
if st.button("Predict"):
    # Prediction
    input_features = np.array([[temperature, humidity, wind_speed]])
    prediction = model.predict(input_features)
    weather_condition = label_encoder.inverse_transform(prediction)[0]
    
    st.write(f"**Predicted Weather Condition:** {weather_condition}")

# Footer
st.markdown("**Developed by Harsh**")
