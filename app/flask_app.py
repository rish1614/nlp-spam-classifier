# app/flask_app.py

from flask import Flask, render_template, request
import sys
import os

# Dynamically add the parent directory to Python's path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from src.predict import SpamPredictor
from src.preprocessing import clean_text

app = Flask(__name__)

# Load the model once when the server starts
model_path = os.path.join(parent_dir, 'models', 'naive_bayes.pkl')
vectorizer_path = os.path.join(parent_dir, 'models', 'bow_vectorizer.pkl')

try:
    predictor = SpamPredictor(model_path, vectorizer_path)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    print("WARNING: Models not found. Please train them first.")

@app.route('/', methods=['GET', 'POST'])
def home():
    if not models_loaded:
        return "Models not found! Please run the training scripts to generate the .pkl files."

    # If the user submitted the form
    if request.method == 'POST':
        user_message = request.form.get('message', '')
        
        if user_message.strip():
            # Get prediction from our src module
            result = predictor.predict(user_message)
            
            # Pass everything to the HTML template
            return render_template(
                'index.html',
                prediction=result['prediction'],
                confidence=round(result['confidence'] * 100, 1),
                is_spam=result['is_spam'],
                original_text=user_message,
                cleaned_text=clean_text(user_message)
            )
            
    # If it's a normal GET request (just loading the page initially)
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, port=5000)
