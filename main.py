import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HemaScan AI | Diagnostic Portal",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LUXURY MEDICAL UI STYLING ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Custom Card Design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161b22;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #8b0000, #ff4b4b);
        color: white;
        border: none;
        padding: 12px;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# --- ASSET LOADING ---
@st.cache_resource
def load_models():
    # Loading the 3 files you exported in cell [145]
    with open("model_ffs.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("ffs_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features


try:
    rf_model, global_scaler, ffs_features = load_models()
except FileNotFoundError:
    st.error("❌ Critical Error: Model files (model_ffs.pkl, scaler.pkl, ffs_features.pkl) not found in directory.")
    st.stop()

# --- SIDEBAR & INFO ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
st.sidebar.title("HemaScan AI v1.0")
st.sidebar.info("""
**Diagnostic Assistant** Using Random Forest + Forward Feature Selection (FFS). 
Targeting 95.97% Accuracy.
""")

# --- MAIN INTERFACE ---
st.title("🩸 Blood Cell Anomaly Detection")
st.markdown("---")

# Organized into 3 logical categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📏 Morphology")
    diameter = st.number_input("Cell Diameter (μm)", 1.0, 25.0, 10.1)
    circularity = st.slider("Circularity Score", 0.0, 1.0, 0.76)
    eccentricity = st.slider("Eccentricity", 0.0, 1.0, 0.36)

with col2:
    st.markdown("### 🧬 Internal Structure")
    chromatin = st.slider("Chromatin Density", 0.0, 1.0, 0.39)
    cyto_ratio = st.slider("Cytoplasm Ratio", 0.0, 1.0, 0.56)
    granularity = st.number_input("Granularity Score", 0.0, 6.0, 1.8)
    lobularity = st.number_input("Lobularity Score", 1.0, 8.0, 1.7)

with col3:
    st.markdown("### 🔬 Optical/Imaging")
    membrane = st.slider("Membrane Smoothness", 0.0, 1.0, 0.84)
    mean_b = st.number_input("Mean Blue Channel", 0, 255, 150)
    stain = st.slider("Stain Intensity", 0.0, 1.0, 0.5)
    magnification = st.selectbox("Magnification (X)", [40, 60, 100], index=2)
    resolution = st.number_input("Resolution (px)", 224, 1024, 336)

# --- PREDICTION LOGIC ---
if st.button("Analyze Cell Sample"):
    # 1. Prepare full 24-feature dummy array for the scaler
    # This matches the 'ss' scaler fitted on 24 columns in cell [107]
    input_full = np.zeros((1, 24))

    # Mapping inputs to their original column indices in X (Cell [101])
    input_full[0, 0] = diameter  # cell_diameter_um
    input_full[0, 2] = chromatin  # chromatin_density
    input_full[0, 3] = cyto_ratio  # cytoplasm_ratio
    input_full[0, 4] = circularity  # circularity
    input_full[0, 5] = eccentricity  # eccentricity
    input_full[0, 6] = granularity  # granularity_score
    input_full[0, 7] = lobularity  # lobularity_score
    input_full[0, 8] = membrane  # membrane_smoothness
    input_full[0, 11] = mean_b  # mean_b
    input_full[0, 12] = stain  # stain_intensity
    input_full[0, 22] = magnification  # magnification_x
    input_full[0, 23] = resolution  # image_resolution_px

    # 2. Scale the input
    scaled_input = global_scaler.transform(input_full)

    # 3. Filter only the 12 features FFS model expects
    # We use the index mapping from your ffs_features.pkl
    # Based on cell [114], these are the specific indices:
    ffs_indices = [0, 2, 3, 4, 5, 6, 7, 8, 11, 12, 22, 23]
    final_features = scaled_input[:, ffs_indices]

    # 4. Prediction
    prediction = rf_model.predict(final_features)[0]
    probability = rf_model.predict_proba(final_features)[0]

    st.markdown("---")

    # --- RESULT DISPLAY ---
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("## ⚠️ DIAGNOSIS: ANOMALY")
            st.write(
                "The cell morphology deviates significantly from healthy reference ranges. Possible indicator of Infection, Anemia, or Leukemia.")
        else:
            st.success("## ✅ DIAGNOSIS: NORMAL")
            st.write("The cell exhibits standard morphological characteristics within normal blood profile limits.")

    with res_col2:
        conf = max(probability) * 100
        st.metric("Model Confidence", f"{conf:.2f}%")
        st.progress(conf / 100)

        if prediction == 1:
            st.warning("Recommendation: Refer to Hematologist for manual smear review.")
        else:
            st.info("Recommendation: Routine follow-up.")

st.markdown("---")
st.caption("Developed for Clinical Research Support | Accuracy 95.97% | Powered by XGBoost & Random Forest")