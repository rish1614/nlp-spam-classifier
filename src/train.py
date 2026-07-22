# src/train.py

import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_model(X_train, y_train, model_type='naive_bayes'):
    """
    Trains a classical ML model on the vectorized text data.
    """
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    else:
        raise ValueError("Unsupported model_type. Use 'naive_bayes', 'logistic_regression', or 'svm'.")
        
    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Saves the trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Loads a trained model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
