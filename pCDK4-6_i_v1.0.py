import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
import joblib
import io

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from PIL import Image

# =========================================================
# Author : Priyanka Solanki
# =========================================================
# Streamlit
logo_url = "https://raw.githubusercontent.com/PriyankaDrugAI/pCDK4-6_i_v1.0/main/logo.png"

st.set_page_config(
    page_title="pCDK4-6-i_v1.0 tool: Predictor of CDK4/6 inhibitors",
    layout="wide",
    page_icon=logo_url
)

st.markdown("""
<style>

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #F1FAF4;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #0B3D2E !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
    font-size: 18px !important;
}

/* RADIO BUTTONS AS CARDS */
.stRadio label {
    background: #145A32 !important;
    padding: 12px 18px !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

.stRadio label:hover {
    background: #1E7D4D !important;
}

/* Sidebar success box */
section[data-testid="stSidebar"] div[data-testid="stAlert"] {
    background-color: #DDF5E3 !important;
    border-radius: 10px;
}

/* Force success box text black */
section[data-testid="stSidebar"] div[data-testid="stAlert"] p,
section[data-testid="stSidebar"] div[data-testid="stAlert"] span,
section[data-testid="stSidebar"] div[data-testid="stAlert"] div {
    color: black !important;
    font-weight: bold;
}

/* Main title */
h1 {
    color: #145A32 !important;
}

/* Headers */
h2, h3 {
    color: #1E5631 !important;
}

/* Expander */
details {
    background-color: #EAF7EE;
    border-radius: 8px;
    padding: 10px;
}

/* Buttons */
.stButton > button:hover {
    background-color: #145A32;
    color: white;
}

.stButton > button:hover {
    background-color: #145A32;
    color: white;
}

</style>
""", unsafe_allow_html=True)
# --------- Utility Functions ---------

def mol_to_array(mol, size=(300, 300)):
    try:
        # Try advanced drawing (best quality)
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(img_data))

    except:
        try:
            # Fallback: simple PIL drawing
            #from rdkit.Chem import Draw
            return Draw.MolToImage(mol, size=size)

        except:
            return None

#from rdkit.Chem import Draw
def get_molecule_image(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/PNG"
    return url
    
def generate_2d_image(smiles, img_size=(300, 300)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=img_size, kekulize=True)
            return img
        else:
            return None
    except:
        return None
        
def pred_label(pred):
    return "### **Active**" if pred == 1 else "### **Inactive**"

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
st.title("pCDK4-6-i_v1.0 tool: Predictor for CDK4/6 inhibitors")

st.markdown(
"""
<h4 style='color:#2E6B4F; font-weight:400;'>
AI-Powered CDK4/6 Inhibitor Prediction Platform
</h4>
<hr>
""",
unsafe_allow_html=True
)

with st.expander("About", expanded=True):
    st.write("""
    This web application predicts:

    1. **Bioactivity Class** → Active / Inactive against CDK4/6 protein
    2. **Probability of Activity**
    3. **Predicted pIC50**

    using machine learning model.
    """)

st.sidebar.image(logo_url)
st.sidebar.success(" **Welcome to pCDK4-6_i_v1.0.** ")

# =========================================================
# SIDEBAR MODE
# =========================================================
mode = st.sidebar.radio(
    "Select Prediction Mode",
    ["Select...", "🧪 Single Molecule Prediction", "📂 Batch Prediction"]
)

if mode == "Select...":
    st.info("Please select a prediction mode from the sidebar.")
    st.stop()

# =========================================================
# SINGLE PREDICTION
# =========================================================
if mode == "🧪 Single Molecule Prediction":

    st.header("Single Molecule Prediction")

    col1, col2 = st.columns([1.2,0.8])

    with col1:
    st.markdown("### Draw Molecule")

    st.markdown("""
    <div style="
        background:white;
        padding:15px;
        border-radius:15px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    ">
    """, unsafe_allow_html=True)

    smile_code = st_ketcher(height=450)

    st.markdown("</div>", unsafe_allow_html=True)

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

          res_col1, res_col2 = st.columns([1,1], gap="small")

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
elif mode == "📂 Batch Prediction":

    st.header("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with 'Smiles' column",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"File loaded successfully! Total molecules: {len(df)}")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

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
     
     priyankasolanki2578@gmail.com
    """)
