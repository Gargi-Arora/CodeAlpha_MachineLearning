"""
app.py
------
Streamlit demo for Disease Prediction from Medical Data.

Run with:
    streamlit run app.py

Lets the user pick a disease (Heart Disease / Diabetes / Breast Cancer),
fill in the relevant clinical inputs, and get a prediction + probability
using the best model trained for that dataset in train.py.
"""

import os
import joblib
import numpy as np
import streamlit as st

ARTIFACT_DIR = "artifacts"

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")
st.title("🩺 Disease Prediction from Medical Data")
st.write("Select a condition, enter patient details, and get a prediction.")

DISEASE_OPTIONS = {
    "Heart Disease": "heart_disease",
    "Diabetes": "diabetes",
    "Breast Cancer": "breast_cancer",
}

choice = st.selectbox("Select condition to check", list(DISEASE_OPTIONS.keys()))
dataset_key = DISEASE_OPTIONS[choice]
model_dir = os.path.join(ARTIFACT_DIR, dataset_key)


@st.cache_resource
def load_artifacts(model_dir):
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    return model, scaler, imputer, feature_names


try:
    model, scaler, imputer, feature_names = load_artifacts(model_dir)
except Exception:
    st.error(
        f"Could not load trained artifacts for {choice} from {model_dir}/. "
        "Run train.py first to generate them."
    )
    st.stop()

st.subheader(f"Enter {choice} details")

# Build input form dynamically from the feature names the model was trained on
user_inputs = {}
cols = st.columns(2)
for i, feat in enumerate(feature_names):
    col = cols[i % 2]
    user_inputs[feat] = col.number_input(feat, value=0.0, format="%.3f")

if st.button("Predict"):
    X_input = np.array([[user_inputs[f] for f in feature_names]])
    X_input = imputer.transform(X_input)
    X_input = scaler.transform(X_input)

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ Prediction: **Disease likely present** ({prob*100:.1f}% probability)")
    else:
        st.success(f"✅ Prediction: **Disease unlikely** ({prob*100:.1f}% probability of disease)")

    st.caption(
        "This is a machine learning demo for educational purposes only and "
        "is not a substitute for professional medical diagnosis."
    )

st.markdown("---")
st.caption("Models: Logistic Regression / Random Forest / SVM / XGBoost (best per dataset selected by F1 score)")
