# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:45:58 2026

@author: DELL
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(page_title="Hierarchical CWA Classifier", layout="wide")

# Load models
rf_stage1 = joblib.load("rf_stage1.pkl")
rf_sch1 = joblib.load("rf_stage2_sch1.pkl")

st.title("Hierarchical CWA Classification App")
st.write("Stage-1: CWA detection | Stage-2: Schedule-1 subclass")

uploaded_file = st.file_uploader(
    "Upload Excel file with Morgan fingerprint columns (FP*)",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    fp_cols = [c for c in df.columns if c.startswith("FP")]

    if len(fp_cols) == 0:
        st.error("No fingerprint columns found.")
        st.stop()

    X = df[fp_cols].values

    # Stage-1
    # Stage-1
        # -------------------------
    # Stage-1 Prediction
    # -------------------------
    
    stage1_pred = rf_stage1.predict(X)
    
    stage1_label_map = {
        0: "Not CWA",
        1: "Dual purpose",
        2: "CWA"
    }
    
    df["Stage1_Label"] = [stage1_label_map[i] for i in stage1_pred]
    
    
    # -------------------------
    # Stage-2 Prediction
    # -------------------------
    
    subclass_map = {
        1: "Nerve",
        2: "Blistering",
        3: "Incapacitating"
    }
    
    # Default = None
    df["Stage2_Subclass"] = "None"
    
    # Only run Stage-2 if Stage-1 predicted CWA
    idx_cwa = np.where(stage1_pred == 2)[0]
    
    if len(idx_cwa) > 0:
        sch1_pred = rf_sch1.predict(X[idx_cwa])
        df.loc[idx_cwa, "Stage2_Subclass"] = [
            subclass_map[i] for i in sch1_pred
        ]
        st.success("Prediction completed.")
        st.dataframe(df.head())
    
        # Download
        output_file = "CWA_predictions.xlsx"
        df.to_excel(output_file, index=False)
    
        with open(output_file, "rb") as f:
            st.download_button(
                "Download Results",
                f,
                file_name=output_file
            )

    
st.markdown("---")
st.subheader("Classify Single Compound (SMILES Input)")
    
smiles_input = st.text_input("Enter SMILES string")
    
if st.button("Predict from SMILES"):
    
            if smiles_input:
    
                mol = Chem.MolFromSmiles(smiles_input)
    
                if mol is None:
                    st.error("Invalid SMILES string.")
                else:
                # Generate Morgan fingerprint (must match training!)
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=2,
                    nBits=1024  # VERY IMPORTANT
                )
    
                fp_array = np.array(fp)
    
                X_single = fp_array.reshape(1, -1)
    
                # -------- Stage-1 ----------
                stage1_pred = rf_stage1.predict(X_single)[0]
    
                stage1_label_map = {
                    0: "Not CWA",
                    1: "Dual purpose",
                    2: "CWA"
                }
    
                stage1_label = stage1_label_map[stage1_pred]
    
                # -------- Stage-2 ----------
                subclass_map = {
                    1: "Nerve",
                    2: "Blistering",
                    3: "Incapacitating"
                }
    
                stage2_label = "None"
    
                if stage1_pred == 2:
                    sch1_pred = rf_sch1.predict(X_single)[0]
                    stage2_label = subclass_map.get(sch1_pred, "Unknown")
    
                # -------- Display Results ----------
                st.success("Prediction Complete")
    
                st.write("### Results")
                st.write("**Stage-1 Classification:**", stage1_label)
                st.write("**Stage-2 Subclass:**", stage2_label)
    
            else:
                st.warning("Please enter a SMILES string.")  