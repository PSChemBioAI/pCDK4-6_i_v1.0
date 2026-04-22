# pCDK4-6_i_v1.0 
>p = predictor,
CDK4-6 = CDK4/6 enzyme,
i = inhibitor,
v1.0 = version 1.0.

<p align="center">
  <img src="https://github.com/PSChemBioAI/pCDK4-6_i_v1.0/blob/main/logo.png?raw=1" width="500">
</p>

**Machine learning-based prediction tool for CDK4/6 inhibitor activity**

It is a <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="Streamlit Logo" width="50"/>-based [Web Application](https://pschembioai-cdk4-6.streamlit.app/). that predict the CDK4/6 inhibitory property (Active or Inactive) of query molecule amd displayed Predicted Probability score along with the Predicted pIC50 values. This tool also allow bacth prediction.

___
This predictive tool was developed as an integral component of the following research manuscript:
>A machine learning approach in combination with a quantum chemistry study to find potent inhibitors against the cell cycle-based cyclin-dependent kinase 4/6 enzyme for anti-cancer treatment.

____

**How to use this?**

The pCDK4-6_i_v1.0 web application can be used by following [This Link](https://pschembioai-cdk4-6.streamlit.app/)

1. Single Molecule Prediction: Enter a SMILES string directly or draw the molecule using the molecular sketcher. The model will analyze the compound and provide the predicted probability of activity and pIC50 value.

2. Batch Prediction: Upload a CSV or Excel file containing the SMILES strings of multiple compounds. The application will process all molecules and return the predicted activity probabilities and pIC50 values.
___

**Example Smiles:**
1. Known CDK4/6 inhibitor: CC(=O)c1c(C)c2cnc(Nc3ccc(N4CCNCC4)cn3)nc2n(C2CCCC2)c1=O
2. Known CDK4/6 Inactive molecule: Cc1cc(C)nc(NC(=S)N2CCN(Cc3ccc(C(F)(F)F)cc3)CC2)c1 
3. Imatinib:
Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1
______
**Applications**
- Virtual Screening of chemical compounds
- Bioactivity prediction in early-stage drug discovery
- Prioritization of lead compounds based on predicted pIC50 values
- Cheminformatics-assisted compound evaluation
---
Bugs: If you encounter any bugs, please report the issue to my mail id priyankasolanki2578@gmail.com
