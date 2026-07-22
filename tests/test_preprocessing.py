# tests/test_preprocessing.py

import pandas as pd
import pytest

# Import the functions from our src module
from src.preprocessing import clean_text, preprocess_dataframe

# --- 1. Tests for clean_text() ---

def test_clean_text_lowercasing():
    """Test that text is properly converted to lowercase."""
    assert clean_text("HELLO World") == "hello world"

def test_clean_text_urls():
    """Test that URLs (http, https, www) are completely removed."""
    text1 = "Check this out: https://github.com/my-repo"
    text2 = "Go to www.google.com right now"
    
    assert clean_text(text1) == "check"
    assert clean_text(text2) == "go right"

def test_clean_text_numbers_and_punctuation():
    """Test that numbers and special characters are stripped out."""
    text = "WINNER!! Call 09061701461 to claim your £1000 prize!!!"
    # 'winner', 'call', 'claim', 'prize' remain. (lemmatizer doesn't change these nouns)
    assert clean_text(text) == "winner call claim prize"

def test_clean_text_stopwords():
    """Test that common English stopwords are removed."""
    text = "this is the best day of my life"
    # 'this', 'is', 'the', 'of', 'my' should be removed
    assert clean_text(text) == "best day life"

def test_clean_text_lemmatization():
    """Test that words are reduced to their root noun forms."""
    text = "the messages are in the boxes"
    # 'messages' -> 'message', 'boxes' -> 'box'
    assert clean_text(text) == "message box"

def test_clean_text_edge_cases():
    """Test how the function handles empty, null, or invalid inputs."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""  # Should return empty string, not crash

# --- 2. Tests for preprocess_dataframe() ---

def test_preprocess_dataframe_logic():
    """Test that the dataframe is processed and empty rows are dropped."""
    
    # Create a mock dataframe covering normal text, a URL-only text, and a number-only text
    mock_data = {
        'message': [
            "Normal messages here",
            "http://only-url.com",  # Should become empty string after cleaning
            "123456",               # Should become empty string after cleaning
            "Good things happen"
        ]
    }
    df = pd.DataFrame(mock_data)
    
    # Run the function
    processed_df = preprocess_dataframe(df, text_col='message')
    
    # Assertions
    assert 'clean_message' in processed_df.columns, "New column was not created"
    
    # Out of 4 original rows, 2 should have been dropped because they became empty
    assert len(processed_df) == 2, f"Expected 2 rows, got {len(processed_df)}"
    
    # Check that lemmatization and cleaning applied correctly to the remaining rows
    assert processed_df.iloc[0]['clean_message'] == "normal message"
    assert processed_df.iloc[1]['clean_message'] == "good thing"
