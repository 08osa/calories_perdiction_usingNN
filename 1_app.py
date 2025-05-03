# Import necessary libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf # Import TensorFlow
from tensorflow import keras # Import Keras

# --- Set Page Config FIRST ---
st.set_page_config(page_title="Calories Predictor (NN)", layout="wide")
# -----------------------------

# --- Configuration: Filenames for the saved NN model and scaler ---
# Ensure these filenames exactly match the files saved after NN training
SCALER_FILENAME = 'scaler_all_models_features.joblib' # Scaler saved from the script that trained all models
MODEL_FILENAME = 'calories_nn_model_final.keras'      # The saved Keras model
# ----------------------------------------------------

# --- Load the Scaler and Model ---
@st.cache_resource # Cache the loaded objects for efficiency
def load_resources(scaler_path, model_path):
    """Loads the scaler and Keras model, handling potential errors."""
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        st.error(f"ERROR: Scaler file '{scaler_path}' not found.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR loading scaler: {e}")
        st.stop()

    try:
        # Use Keras function to load the model
        model = keras.models.load_model(model_path)
        print(f"Keras model loaded from {model_path}")
    except FileNotFoundError:
        st.error(f"ERROR: Model file '{model_path}' not found.")
        st.stop()
    except Exception as e:
        # Keras loading can sometimes give specific errors (e.g., HDF5)
        st.error(f"ERROR loading Keras model: {e}")
        st.stop()
    return scaler, model

# Load the resources
scaler, model = load_resources(SCALER_FILENAME, MODEL_FILENAME)

# --- Define Feature Names (Crucial: Must match training order!) ---
# This order MUST match the columns in 'X' DataFrame from your training script
feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# --- Streamlit User Interface ---
st.title('🔥 Calories Burnt Prediction (Neural Network) 🔥')
st.markdown('Enter your details and workout metrics below to predict the calories burnt using our NN model.')
st.divider()

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 User Info")
    gender_input = st.radio("Gender", ('Male', 'Female'), key='gender', horizontal=True)
    age_input = st.number_input("Age (years)", min_value=20, max_value=79, value=25, step=1, key='age')
    height_input = st.number_input("Height (cm)", min_value=123.0, max_value=222.0, value=170.0, step=0.5, key='height')
    weight_input = st.number_input("Weight (kg)", min_value=36.0, max_value=132.0, value=70.0, step=0.5, key='weight')

with col2:
    st.subheader("🏋️ Workout Metrics")
    duration_input = st.number_input("Duration (minutes)", min_value=1.0, max_value=30.0, value=15.0, step=1.0, key='duration')
    heart_rate_input = st.number_input("Average Heart Rate (bpm)", min_value=67.0, max_value=128.0, value=100.0, step=1.0, key='heart_rate')
    body_temp_input = st.number_input("Average Body Temperature (°C)", min_value=37.1, max_value=41.5, value=38.0, step=0.1, key='body_temp')

st.divider()

# --- Prediction Logic ---
col_button1, col_button2, col_button3 = st.columns([2,1,2])
with col_button2:
    predict_button = st.button('Predict Calories Burnt', key='predict_button', use_container_width=True)

if predict_button:
    gender_encoded = 0 if gender_input == 'Male' else 1

    input_data = pd.DataFrame([[
        gender_encoded,
        age_input,
        height_input,
        weight_input,
        duration_input,
        heart_rate_input,
        body_temp_input
    ]], columns=feature_names)

    try:
        input_data_scaled = scaler.transform(input_data)
        # Keras model expects numpy array usually, though it might handle DataFrame
        input_data_scaled_np = np.array(input_data_scaled)

        # Make Prediction using the loaded Keras model
        prediction = model.predict(input_data_scaled_np)
        # Keras predict might return a 2D array [[value]], get the scalar value
        predicted_calories = prediction[0][0]

        st.success(f"**Predicted Calories Burnt: {predicted_calories:.2f} kcal**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the input data is valid and the model/scaler loaded correctly.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info(f"""
This app predicts calories burnt during exercise using a pre-trained Neural Network model (Keras/TensorFlow).

**Model Details:**
- **Type:** Neural Network (MLP)
- **Features Used:** {len(feature_names)}
({', '.join(feature_names)})

Enter your details on the main page and click 'Predict'.
""")
