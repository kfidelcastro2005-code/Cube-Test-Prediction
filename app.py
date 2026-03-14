import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Concrete Strength Predictor + Cube Test", layout="wide")
st.title("🧱 Concrete Compressive Strength ANN Predictor + Cube Failure Test")
st.markdown("**Predict with ANN → Test actual cube (IS 516) → Check accuracy**")

# Load model
model = load_model('ann_concrete_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Tab 1: ANN Prediction
tab1, tab2 = st.tabs(["ANN Prediction", "Cube Test Calculator & Accuracy"])

with tab1:
    st.header("1. Predict Strength using ANN")
    col1, col2 = st.columns(2)
    with col1:
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 300.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 0.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 0.0)
        water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0)
    with col2:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 0.0)
        coarse = st.number_input("Coarse Aggregate (kg/m³)", 500.0, 1500.0, 1000.0)
        fine = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1200.0, 800.0)
        age = st.number_input("Age (days)", 1, 365, 28)

    if st.button("Predict Strength"):
        input_data = np.array([[cement, slag, flyash, water, superplasticizer, coarse, fine, age]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0][0]
        st.success(f"**Predicted Compressive Strength: {pred:.2f} MPa**")
        st.session_state['pred'] = pred   # save for later comparison

# Tab 2: Cube Test + Accuracy
with tab2:
    st.header("2. Actual Cube Test (Failure Load + Weight)")
    st.info("Standard 150 mm cube as per IS 516. Weight used for density check.")

    cube_size = st.selectbox("Cube Size (mm)", [100, 150], index=1)
    weight_kg = st.number_input("Weight of Cube (kg)", 1.0, 15.0, 8.5, step=0.01)
    failure_load_kn = st.number_input("Failure Load (kN)", 100.0, 3000.0, 600.0)

    if st.button("Calculate Actual Strength & Accuracy"):
        side_mm = cube_size
        area_mm2 = side_mm * side_mm
        actual_strength = (failure_load_kn * 1000) / area_mm2   # MPa

        # Density
        volume_m3 = (side_mm/1000)**3
        density = weight_kg / volume_m3

        st.subheader("Results")
        st.metric("Actual Compressive Strength", f"{actual_strength:.2f} MPa")
        st.metric("Density", f"{density:.0f} kg/m³")

        if 'pred' in st.session_state:
            pred = st.session_state['pred']
            error = abs(pred - actual_strength) / actual_strength * 100
            st.metric("Prediction Accuracy", f"{100 - error:.1f}%", delta=f"Error: {error:.1f}%")

            if actual_strength >= 0.85 * pred and actual_strength <= 1.15 * pred:
                st.success("✅ Cube PASSED - Prediction matches test within ±15% (IS code tolerance)")
            else:
                st.error("❌ Cube FAILED or prediction inaccurate")

            # Pass/Fail check (example M25 concrete)
            if actual_strength < 25:
                st.warning("Cube FAILED required strength (example M25)")
        else:
            st.warning("First predict with ANN in Tab 1")
