import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Bioactivity Predictor",
    layout="wide"
)

st.title("Bioactivity Prediction Web App")
st.write("Predict activity class and pIC50 from SMILES")

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    clf_model = joblib.load("xgb_model.pkl")
    reg_model = joblib.load("xgb_fp_reg_model.pkl")
    return clf_model, reg_model

clf_model, reg_model = load_models()

# -------------------------------
# FINGERPRINT GENERATOR
# -------------------------------
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

# -------------------------------
# FEATURE FUNCTION
# -------------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    fp = fpg.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    img = Draw.MolToImage(mol, size=(300, 300))

    return arr.reshape(1, -1), img

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_smiles(smiles):
    fp, img = smiles_to_fp(smiles)

    if fp is None:
        return None

    clf_pred = clf_model.predict(fp)[0]
    clf_prob = clf_model.predict_proba(fp)[0][1]
    reg_pred = reg_model.predict(fp)[0]

    return clf_pred, clf_prob, reg_pred, img

# -------------------------------
# SIDEBAR
# -------------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Molecule Prediction", "Batch Prediction"]
)

# =====================================================
# SINGLE MOLECULE PREDICTION
# =====================================================
if mode == "Single Molecule Prediction":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Draw Molecule")
        smiles_drawn = st_ketcher()

    with col2:
        smiles_input = st.text_input(
            "Enter SMILES",
            value=smiles_drawn if smiles_drawn else ""
        )

    if st.button("Predict"):

        result = predict_smiles(smiles_input)

        if result is None:
            st.error("Invalid SMILES")
        else:
            clf_pred, clf_prob, reg_pred, img = result

            label = "Active" if clf_pred == 1 else "Inactive"

            colA, colB = st.columns(2)

            with colA:
                st.image(img, caption="Query Molecule")

            with colB:
                st.subheader("Prediction Results")
                st.success(f"Predicted Class: {label}")
                st.write(f"Probability of Active: {clf_prob:.3f}")
                st.write(f"Predicted pIC50: {reg_pred:.3f}")

# =====================================================
# BATCH PREDICTION
# =====================================================
elif mode == "Batch Prediction":

    uploaded_file = st.file_uploader(
        "Upload CSV with 'Smiles' column",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "Smiles" not in df.columns:
            st.error("CSV must contain 'Smiles' column")
        else:

            classes = []
            probs = []
            pic50s = []

            for smiles in df["Smiles"]:

                result = predict_smiles(smiles)

                if result is None:
                    classes.append("Invalid")
                    probs.append(None)
                    pic50s.append(None)
                else:
                    clf_pred, clf_prob, reg_pred, img = result
                    label = "Active" if clf_pred == 1 else "Inactive"

                    classes.append(label)
                    probs.append(clf_prob)
                    pic50s.append(reg_pred)

            df["Predicted_Class"] = classes
            df["Probability_Active"] = probs
            df["Predicted_pIC50"] = pic50s

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "prediction_results.csv",
                "text/csv"
            )
