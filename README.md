# 🎙️ Grammar Scoring Engine 🧠✅

A multimodal AI application that predicts the **grammar quality score** of spoken language. Upload your audio file, and the system extracts audio features using `librosa`, evaluates grammar fluency, and returns a score — all in a visually interactive Streamlit app.

---

## 📌 Features

- 🎧 **Audio Input** (`.wav`, `.mp3`) — upload speech recordings
- 🧠 **Grammar Scoring** — ML model trained on Mozilla Common Voice transcriptions
- 📊 **Feature Engineering** — MFCCs, spectral contrast, zero-crossing rate
- 📈 **Visualizations** — waveform and MFCC plots with `matplotlib` and `librosa`
- 📁 **Downloadable Report** — CSV file with extracted features and predicted score

----
## Dataset Link
https://www.kaggle.com/datasets/mozillaorg/common-voice?select=cv-valid-train

---

## 🛠️ Tech Stack

| Category         | Tools Used                        |
|------------------|------------------------------------|
| Interface        | `Streamlit`                        |
| Audio Processing | `librosa`, `numpy`, `matplotlib`   |
| ML Model         | `scikit-learn`, `joblib`           |
| Feature Scaling  | `StandardScaler` (from `sklearn`)  |

---

## 🧪 How It Works

1. **Audio Upload**: User uploads a `.wav` or `.mp3` file.
2. **Feature Extraction**:
    - MFCCs (mean & std)
    - Spectral Contrast (mean & std)
    - Zero Crossing Rate
3. **Prediction**:
    - Scaled features are passed to a trained ML model.
    - A grammar score is predicted (range: `0.0` to `7.0`).
4. **Output**:
    - Display of score
    - Waveform and MFCC visualizations
    - Option to download full feature report (CSV)

---

## 📂 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `grammar_model.joblib` | Pre-trained grammar scoring model |
| `scaler.joblib` | Feature scaler used during training |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation file |

---

## 📦 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/grammar-scoring-engine.git
   cd grammar-scoring-engine
