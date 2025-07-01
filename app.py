import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import librosa
import librosa.display
import tempfile
import streamlit as st
import io
import os

# ========== Load Model & Scaler ==========
model = joblib.load('grammar_model.joblib')
scaler = joblib.load('scaler.joblib')

# ========== Initialize App ==========
st.title("Welcome to the Grammar Scoring Engine 🧠✅ ")

# ========== Initialize session state for report tracking ==========
if 'report_history' not in st.session_state:
    st.session_state.report_history = []

# ========== File Upload ==========
uploaded_file = st.file_uploader("Upload your audio file 🔊", type=['wav', 'mp3'])

if uploaded_file:

    st.audio(uploaded_file, format='audio/wav')
    filename = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name

    # ========== Feature Extraction ==========
    y, sr = librosa.load(audio_path, sr=22050)
    features = {"filename": filename}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # ========== Prediction ==========
    input_data = pd.DataFrame([features])[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_data)
    score = model.predict(input_scaled)[0]

    features['predicted_score'] = round(score, 4)
    st.session_state.report_history.append(features)

    st.metric("Grammar Score", f"{score:.2f}/7.0")

    # ========== Waveform ==========
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
    ax.set_title("Waveform")
    st.pyplot(fig)

    # ========== MFCCs ==========
    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc_img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    fig.colorbar(mfcc_img, ax=ax, format="%+2.f")
    ax.set_title("MFCCs")
    st.pyplot(fig)

    # ========== Download Report ==========
    report_df = pd.DataFrame(st.session_state.report_history)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Score Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="grammar_score_report.csv",
        mime="text/csv"
    )


