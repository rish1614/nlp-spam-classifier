# app/app.py

import streamlit as st
import sys
import os

# Dynamically add the parent directory to Python's path
# This allows us to import the 'src' package from the 'app' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from src.predict import SpamPredictor
from src.preprocessing import clean_text

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spam Classifier NLP",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL (Cached) ---
# @st.cache_resource ensures the model is only loaded into memory once,
# not every time the user clicks a button.
@st.cache_resource
def load_predictor():
    model_path = os.path.join(parent_dir, 'models', 'naive_bayes.pkl')
    vectorizer_path = os.path.join(parent_dir, 'models', 'bow_vectorizer.pkl')
    
    # Check if the models have been generated yet
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None
        
    return SpamPredictor(model_path, vectorizer_path)

predictor = load_predictor()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ NLP Spam Filter")
    st.info(
        "This is an end-to-end Natural Language Processing (NLP) application "
        "built to classify text messages and emails."
    )
    st.markdown(
        """
        **Under the Hood:**
        1. Text Cleaning & Lemmatization
        2. Bag of Words (CountVectorizer)
        3. Multinomial Naive Bayes
        """
    )
    st.divider()
    st.caption("Built with Python, Scikit-Learn, and Streamlit.")

# --- MAIN APP UI ---
st.title("✉️ Spam Message Detector")
st.write("Paste an email or text message below to see if the model flags it as spam.")

# Graceful error handling if models aren't trained yet
if predictor is None:
    st.error(
        "⚠️ **Models not found!**\n\n"
        "Please run your training notebooks (or `src/train.py`) first to generate "
        "`naive_bayes.pkl` and `bow_vectorizer.pkl` inside the `models/` directory."
    )
    st.stop()

# Text Input Area
user_input = st.text_area("Message Content:", height=150, placeholder="Example: Congratulations! You've won a free $1000 gift card. Click here to claim...")

# Analyze Button
if st.button("Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Call the predict function from our src module
            result = predictor.predict(user_input)
            
            prediction = result['prediction']
            confidence = result['confidence'] * 100
            
            st.divider()
            st.subheader("Result")
            
            # Display visually distinct results based on the prediction
            if prediction == "Spam":
                st.error("🚨 **SPAM DETECTED**")
                
                # Use Streamlit metrics for a nice UI presentation
                col1, col2 = st.columns(2)
                col1.metric("Classification", "Spam", "-")
                col2.metric("Confidence Score", f"{confidence:.1f}%")
                
            else:
                st.success("✅ **SAFE (HAM)**")
                
                col1, col2 = st.columns(2)
                col1.metric("Classification", "Safe", "+")
                col2.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Educational Expander: Show the user what the model *actually* saw
            with st.expander("🔍 See what the model saw (Preprocessing)"):
                st.write("**1. Original Text:**")
                st.write(user_input)
                
                st.write("**2. After Cleaning & Lemmatization:**")
                cleaned = clean_text(user_input)
                st.code(cleaned if cleaned else "[No valid text left after cleaning]")
                
                st.caption("Notice how punctuation, uppercase letters, and 'stopwords' (like 'the', 'is') are removed, and words are converted to their root form.")
