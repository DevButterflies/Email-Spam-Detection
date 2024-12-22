import pytest
import pandas as pd
import numpy as np
from src.preprocess import EmailPreprocessor
import tempfile
import os

@pytest.fixture
def preprocessor():
    return EmailPreprocessor()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
    'Messages': [
        'Hello, how are you?',  # Existing 'ham' message
        'WIN FREE MONEY NOW!!!',  # Existing 'spam' message
        'Meeting at 2pm tomorrow',  # Existing 'ham' message
        'CLICK HERE for amazing deals!',  # Existing 'spam' message
        'Hi John, the report is ready',  # Existing 'ham' message

        # Augmented 'ham' messages
        'Hey, can we reschedule our meeting?',  
        'Hope everything is going well with your project.',
        'Let me know when you are free to chat.',
        'Looking forward to catching up later.',
        'I’m almost done with the updates, just need a few more minutes.',

        # Augmented 'spam' messages
        'Limited time offer! Buy now and get 50% off!',  
        'Congratulations, you’ve won a $1000 gift card!',
        'CLICK NOW to claim your free vacation!',
        'Earn money from home with no experience required.',
        'This is your final notice about your account being suspended. Act now!'

    ],
    'Category': [
        'ham',  # Corresponding categories for the existing messages
        'spam',
        'ham',
        'spam',
        'ham',

        # Corresponding categories for the augmented 'ham' messages
        'ham',
        'ham',
        'ham',
        'ham',
        'ham',

        # Corresponding categories for the augmented 'spam' messages
        'spam',
        'spam',
        'spam',
        'spam',
        'spam'
    ]})

#(checked test)
def test_clean_text(preprocessor):
    """Test text cleaning functionality."""
    # Test basic cleaning
    text = "Hello! How are you? 123"
    cleaned = preprocessor.clean_text(text)
    assert isinstance(cleaned, str)
    assert "123" not in cleaned
    assert cleaned.islower()
    
    # Test email removal
    text = "Contact us at test@example.com"
    cleaned = preprocessor.clean_text(text)
    assert "@" not in cleaned
    
    # Test URL removal
    text = "Visit http://example.com for more"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    
    # Test handling non-string input
    cleaned = preprocessor.clean_text(123)
    assert cleaned == ""

def test_prepare_data(preprocessor, sample_data):
    """Test data preparation functionality."""
    # Test with DataFrame input
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        sample_data, test_size=0.4, random_state=42
    )
    
    # Check shapes
    assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    
    # Check binary labels
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})
    
    # Test with file input
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        sample_data.to_csv(f.name, index=False)
        f.close()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(f.name)
        os.unlink(f.name)
    
    # Test error handling
    with pytest.raises(ValueError):
        bad_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        preprocessor.prepare_data(bad_data)
    
    with pytest.raises(FileNotFoundError):
        preprocessor.prepare_data('nonexistent.csv')

def test_transform_text(preprocessor):
    """Test text transformation functionality."""
    # First prepare the vectorizer with some data
    sample_text = pd.read_csv(r"C:\Users\atef nasri\Documents\Projects\solo_Projects\Spam_Detection\spam-email-detection-1\data\spam mail.csv")
    preprocessor.vectorizer.fit(sample_text)
    
    # Test single text transformation
    text = "Hello, this is a test message!"
    transformed = preprocessor.transform_text(text)
    
    assert hasattr(transformed, 'shape')
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(preprocessor.vectorizer.get_feature_names_out())

def test_save_load(preprocessor, tmp_path):
    """Test saving and loading functionality."""
    # Save preprocessor
    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(save_path)
    
    # Load preprocessor
    loaded_preprocessor = EmailPreprocessor.load(save_path)
    
    # Check that the loaded preprocessor works
    text = "Test message"
    original_cleaned = preprocessor.clean_text(text)
    loaded_cleaned = loaded_preprocessor.clean_text(text)
    
    assert original_cleaned == loaded_cleaned
