import streamlit as st
import os
import sys
from pathlib import Path

# --- Path Fix: Ensure the UI can find your src/ modules ---
# This adds the project root to the python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer

st.set_page_config(page_title="Sherlock Predictor", page_icon="🕵️")

# --- Resource Loading ---
@st.cache_resource
def load_all():
    # Use absolute paths based on project_root to avoid FileNotFoundError
    # Verify if your folder is named 'model' or 'models' in your file explorer!
    model_path = project_root / "data" / "model" / "model.json" 
    vocab_path = project_root / "data" / "model" /"vocab.json"
    
    norm = Normalizer()
    # Ensure these parameters match what you used in main.py
    model = NGramModel(unk_threshold=3, ngram_order=4)
    
    if not model_path.exists():
        st.error(f"Could not find model at {model_path}. Please check your folder name.")
        st.stop()
        
    model.load_model(str(model_path), str(vocab_path))
    return Predictor(model, norm)

# Initialize Predictor
predictor = load_all()

# --- UI Layout ---
st.title("🕵️ Sherlock Text Predictor")
st.markdown("This predictor uses an **N-Gram model** trained on the Sherlock Holmes corpus.")

input_text = st.text_input("Enter your phrase:", "She is always the")

col1, col2 = st.columns([1, 1])
with col1:
    k_val = st.slider("Number of results (K)", 1, 10, 3)

if input_text:
    with st.spinner('Thinking...'):
        results = predictor.predict_next(input_text, k=k_val)
    
    if results:
        st.subheader("Top Suggestions:")
        # Display predictions as clickable buttons
        cols = st.columns(len(results))
        for i, word in enumerate(results):
            cols[i].button(word)
    else:
        st.warning("No predictions found for this context.")

# --- Extra Credit: Smoothing Toggle ---
st.sidebar.title("Settings")
smoothing_enabled = st.sidebar.toggle("Enable Smoothing (Extra Credit)")
# Note: You'll need to pass this flag to your predictor/model if implemented!