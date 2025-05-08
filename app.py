import streamlit as st
import joblib
import pandas as pd
import numpy as np
import language_tool_python

# Load model
model = joblib.load("random_forest_model_250.pkl")

# Grammar analysis function
tool = language_tool_python.LanguageTool('en-US')

def extract_text_features(text):
    matches = tool.check(text)
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    return {
        'grammar_errors': len(matches),
        'error_rate': len(matches)/len(words) if words else 0,
        'word_count': len(words),
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words))/len(words) if words else 0,
        'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        'repetition_score': len(words)/len(set(words)) if words else 1,
        'repetitive_phrases': count_repetitive_phrases(text)
    }

def count_repetitive_phrases(text, min_repeats=3):
    words = text.split()
    repeats = 0
    for i in range(len(words) - min_repeats):
        if words[i] == words[i+1] == words[i+2]:
            repeats += 1
    return repeats

# UI
st.title("Grammar Scoring App")
st.write("Enter your transcription text below to predict the grammar quality score.")

user_input = st.text_area("Enter transcription text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        features = extract_text_features(user_input)
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        st.success(f"Predicted Score: {prediction:.2f}")
        st.json(features)
