"""
app.py
------
Streamlit demo for Speech Emotion Recognition.

Run with:
    streamlit run app.py

Loads the trained model + scaler + label encoder from ./artifacts/
and lets the user upload a .wav file to get a predicted emotion with
confidence scores.
"""

import os
import joblib
import numpy as np
import streamlit as st
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

from feature_extraction import extract_features, SAMPLE_RATE

ARTIFACT_DIR = "artifacts"

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a short audio clip (.wav) and the model will predict the emotion.")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "best_model.keras"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    return model, scaler, le


try:
    model, scaler, le = load_artifacts()
except Exception as e:
    st.error(
        "Could not load trained artifacts from ./artifacts/. "
        "Run train.py first to generate best_model.keras, scaler.pkl, "
        "and label_encoder.pkl."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing audio..."):
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)

        # Waveform plot
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(y, color="#4C72B0")
        ax.set_title("Waveform")
        ax.set_xticks([])
        st.pyplot(fig)

        feats = extract_features(y, sr).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        feats_cnn = feats_scaled.reshape(1, feats_scaled.shape[1], 1)

        probs = model.predict(feats_cnn)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        confidence = probs[pred_idx] * 100

    st.success(f"**Predicted emotion: {pred_label.upper()}**  ({confidence:.1f}% confidence)")

    st.subheader("Confidence per emotion")
    prob_dict = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    st.bar_chart(prob_dict)

st.markdown("---")
st.caption("Model: CNN + BiLSTM hybrid trained on RAVDESS | Features: MFCC, Chroma, Mel, ZCR, Spectral Contrast")
