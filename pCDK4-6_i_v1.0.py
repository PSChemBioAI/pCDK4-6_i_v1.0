import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
import joblib
import io

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Draw
from PIL import Image

# =========================================================
# Author : Priyanka Solanki
# =========================================================
logo_url = "images/logo.png"
st.image(logo_url)

st.set_page_config(
    page_title="Bioactivity Predictor",
    layout="wide",
    page_icon=logo_url
)

st.markdown("""
<style>
section[data-testid="stSidebar"] label {
    font-size: 20px !important;
    font-weight: bold !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    clf_model = joblib.load("xgb_model.pkl")
    reg_model = joblib.load("xgb_fp_reg_model.pkl")
    return clf_model, reg_model

clf_model, reg_model = load_models()

# =========================================================
# FINGERPRINT GENERATOR
# =========================================================
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def generate_molecule_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            return img
        return None
    except:
        return None

def predict_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = fpg.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_array = arr.reshape(1, -1)

    clf_pred = clf_model.predict(fp_array)[0]
    clf_prob = clf_model.predict_proba(fp_array)[0][1]
    reg_pred = reg_model.predict(fp_array)[0]

    return clf_pred, clf_prob, reg_pred

# =========================================================
# UI HEADER
# =========================================================
st.title("Bioactivity Prediction Web App")

with st.expander("About", expanded=True):
    st.write("""
    This web application predicts:

    1. **Bioactivity Class** → Active / Inactive  
    2. **Probability of Activity**
    3. **Predicted pIC50**

    using machine learning models.
    """)

st.sidebar.image(logo_url)
st.sidebar.success("Welcome to Bioactivity Predictor")

# =========================================================
# SIDEBAR MODE
# =========================================================
mode = st.sidebar.radio(
    "Select Prediction Mode",
    ["Select...", "Single Molecule Prediction", "Batch Prediction"]
)

if mode == "Select...":
    st.info("Please select a prediction mode from the sidebar.")
    st.stop()

# =========================================================
# SINGLE PREDICTION
# =========================================================
if mode == "Single Molecule Prediction":

    st.header("Single Molecule Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Draw Molecule")
        smile_code = st_ketcher()

    with col2:
        st.markdown("### SMILES Input")
        smiles_input = st.text_input(
            "Enter or edit SMILES:",
            value=smile_code if smile_code else ""
        )

    if smiles_input:
        with st.spinner("Predicting..."):
            result = predict_smiles(smiles_input)

        if result is None:
            st.error("Invalid SMILES!")
        else:
            clf_pred, clf_prob, reg_pred = result
            label = "Active" if clf_pred == 1 else "Inactive"

            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                img = generate_molecule_image(smiles_input)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.warning("Unable to generate molecule image.")

            with res_col2:
                st.markdown(
                    f"""
                    <div style="
                        font-size:42px;
                        font-weight:700;
                        text-align:center;
                        color:{'green' if clf_pred == 1 else 'red'};
                    ">
                    {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="text-align:center; font-size:18px; margin-top:15px;">
                    <b>Probability (Active)</b><br>
                    {clf_prob:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="text-align:center; font-size:18px; margin-top:15px;">
                    <b>Predicted pIC50</b><br>
                    {reg_pred:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# BATCH PREDICTION
# =========================================================
elif mode == "Batch Prediction":

    st.header("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file with 'Smiles' column",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "Smiles" not in df.columns:
            st.error("File must contain a 'Smiles' column")
            st.stop()

        predictions = []
        probabilities = []
        pic50s = []
        labels = []

        with st.spinner("Running predictions..."):
            for smi in df["Smiles"]:
                result = predict_smiles(smi)

                if result is None:
                    predictions.append(None)
                    probabilities.append(None)
                    pic50s.append(None)
                    labels.append("Invalid SMILES")
                else:
                    clf_pred, clf_prob, reg_pred = result
                    predictions.append(clf_pred)
                    probabilities.append(clf_prob)
                    pic50s.append(reg_pred)
                    labels.append("Active" if clf_pred == 1 else "Inactive")

        df["Prediction"] = predictions
        df["Probability"] = probabilities
        df["Predicted_pIC50"] = pic50s
        df["Label"] = labels

        st.success("Batch prediction completed!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

# =========================================================
# CONTACT
# =========================================================
with st.expander("Contact"):
    st.write("""
    **Priyanka Solanki**  
     https://github.com/PriyankaDrugAI
    """)
