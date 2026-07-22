# src/features.py

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vectorizer(train_texts, method='tfidf', max_features=5000):
    """
    Initializes and fits a vectorizer on the training data.
    """
    if method == 'bow':
        vectorizer = CountVectorizer(max_features=max_features)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    else:
        raise ValueError("Method must be 'bow' or 'tfidf'")
        
    X_train_vec = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train_vec

def save_vectorizer(vectorizer, filepath):
    """Saves the fitted vectorizer to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(filepath):
    """Loads a fitted vectorizer from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
