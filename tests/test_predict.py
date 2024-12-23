"""import pytest
import joblib
import os
from src.predict import SpamPredictor, load_predictor
from src.model import SpamDetectionModel
from src.preprocess import EmailPreprocessor

@pytest.fixture
def setup_predictor(tmp_path):
    #Create a test predictor with dummy model and preprocessor.
    # Create and save a dummy model
    model = SpamDetectionModel('naive_bayes')
    preprocessor = EmailPreprocessor()
    
    # Create dummy data and train model
    X = preprocessor.vectorizer.fit_transform(['test message'])
    y = [0]
    model.train(X, y)
    
    # Save model and preprocessor
    model_path = tmp_path / "test_model.joblib"
    preprocessor_path = tmp_path / "test_preprocessor.joblib"
    
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    return SpamPredictor(model_path, preprocessor_path)

def test_predictor_initialization(tmp_path):
    Test predictor initialization and error handling.
    # Test with nonexistent files
    with pytest.raises(FileNotFoundError):
        SpamPredictor("nonexistent_model.joblib", "nonexistent_preprocessor.joblib")
    
    # Create valid files
    model = SpamDetectionModel()
    preprocessor = EmailPreprocessor()
    
    model_path = tmp_path / "model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Test successful initialization
    predictor = SpamPredictor(model_path, preprocessor_path)
    assert isinstance(predictor.model, SpamDetectionModel)
    assert isinstance(predictor.preprocessor, EmailPreprocessor)

def test_predict_message(setup_predictor):
    #Test single message prediction.
    # Test ham message
    result = setup_predictor.predict_message("Hello, how are you?")
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'is_spam' in result
    assert result['prediction'] in ['spam', 'ham']
    assert isinstance(result['is_spam'], bool)
    
    # Test spam message
    result = setup_predictor.predict_message("CONGRATULATIONS! You've won $1,000,000!")
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'is_spam' in result
    
    # Test empty message
    result = setup_predictor.predict_message("")
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'is_spam' in result

def test_predict_batch(setup_predictor):
    #Test batch prediction functionality.
    messages = [
        "Hello, how are you?",
        "CONGRATULATIONS! You've won $1,000,000!",
        "Meeting at 2pm tomorrow",
        ""  # Empty message
    ]
    
    results = setup_predictor.predict_batch(messages)
    
    assert len(results) == len(messages)
    for result in results:
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'is_spam' in result
        assert result['prediction'] in ['spam', 'ham']
        assert isinstance(result['is_spam'], bool)

def test_load_predictor(tmp_path):
    #Test predictor loading functionality.
    # Create models directory
    models_dir = tmp_path / "models"
    os.makedirs(models_dir)
    
    # Test with no models
    with pytest.raises(FileNotFoundError):
        load_predictor(models_dir)
    
    # Create and save test model and preprocessor
    model = SpamDetectionModel()
    preprocessor = EmailPreprocessor()
    
    model_path = models_dir / "spam_model_naive_bayes.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"
    
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Test successful loading
    predictor = load_predictor(models_dir)
    assert isinstance(predictor, SpamPredictor)
    assert isinstance(predictor.model, SpamDetectionModel)
    assert isinstance(predictor.preprocessor, EmailPreprocessor)"""
