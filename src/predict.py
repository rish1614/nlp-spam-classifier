# src/predict.py

import numpy as np
from .preprocessing import clean_text
from .features import load_vectorizer
from .train import load_model

class SpamPredictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initializes the predictor by loading the saved model and vectorizer.
        """
        self.model = load_model(model_path)
        self.vectorizer = load_vectorizer(vectorizer_path)
        
    def predict(self, raw_text: str) -> dict:
        """
        Takes raw text, cleans it, vectorizes it, and returns a prediction and confidence score.
        """
        # 1. Preprocess
        cleaned_text = clean_text(raw_text)
        
        # Handle edge case where cleaning removes all text
        if not cleaned_text.strip():
            return {
                "prediction": "Ham", # Default safe prediction
                "confidence": 0.0,
                "is_spam": False
            }
            
        # 2. Vectorize (must be in an iterable/list format for sklearn)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        # 3. Predict
        pred_label = self.model.predict(vectorized_text)[0]
        
        # Try to get probabilities if the model supports it
        try:
            probabilities = self.model.predict_proba(vectorized_text)[0]
            confidence = float(np.max(probabilities))
        except AttributeError:
            # Fallback if model (like a hard SVM) doesn't output probabilities
            confidence = 1.0 
            
        is_spam = bool(pred_label == 1)
        
        return {
            "prediction": "Spam" if is_spam else "Ham",
            "confidence": round(confidence, 4),
            "is_spam": is_spam
        }

# Quick test execution block
if __name__ == "__main__":
    # Example usage (assuming models are trained and saved in the expected paths)
    import os
    model_path = '../models/naive_bayes.pkl'
    vec_path = '../models/bow_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        predictor = SpamPredictor(model_path, vec_path)
        
        test_messages = [
            "Hey man, are we still on for the meeting tomorrow at 10?",
            "CONGRATULATIONS! You have won a $1000 Walmart gift card. Click here to claim NOW!"
        ]
        
        for msg in test_messages:
            result = predictor.predict(msg)
            print(f"Message: '{msg}'\nResult: {result}\n")
    else:
        print("Models not found. Train and save models first using the notebooks or train.py.")
