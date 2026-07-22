# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Cleans raw text by lowercasing, removing URLs, punctuation, 
    stopwords, and applying lemmatization.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 3. Remove punctuation and numbers (keep only alphabets)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_dataframe(df, text_col='message'):
    """
    Applies the clean_text function to a specific column in a DataFrame.
    """
    df['clean_message'] = df[text_col].apply(clean_text)
    # Drop rows that became empty after cleaning
    df = df[df['clean_message'].str.strip() != '']
    return df
