import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.model import SpamDetectionModel, ModelEvaluator
import tempfile
import os

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def model():
    """Create a default model instance."""
    return SpamDetectionModel()

def test_model_initialization():
    """Test model initialization with different types."""
    # Test valid model types
    for model_type in SpamDetectionModel.AVAILABLE_MODELS.keys():
        model = SpamDetectionModel(model_type)
        assert model.model_type == model_type
    
    # Test invalid model type
    with pytest.raises(ValueError):
        SpamDetectionModel('invalid_model')

def test_model_training(model, sample_data):
    """Test model training functionality."""
    X, y = sample_data
    model.train(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (int, np.integer, bool)) for pred in predictions)
    assert set(predictions).issubset({0, 1})

def test_model_evaluation(model, sample_data):
    """Test model evaluation functionality."""
    X, y = sample_data
    model.train(X, y)
    
    metrics = model.evaluate(X, y)
    
    # Check metric structure
    assert 'accuracy' in metrics
    assert 'classification_report' in metrics
    assert 'class_metrics' in metrics
    
    # Check accuracy bounds
    assert 0 <= metrics['accuracy'] <= 1
    
    # Check class metrics
    for class_name in ['ham', 'spam']:
        assert class_name in metrics['class_metrics']
        class_metrics = metrics['class_metrics'][class_name]
        for metric in ['precision', 'recall', 'f1']:
            assert metric in class_metrics
            assert 0 <= class_metrics[metric] <= 1

def test_model_save_load(model, sample_data, tmp_path):
    """Test model saving and loading functionality."""
    X, y = sample_data
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    model.save(model_path)
    
    # Load model
    loaded_model = SpamDetectionModel.load(model_path, model.model_type)
    
    # Compare predictions
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(original_preds, loaded_preds)

def test_model_evaluator(sample_data):
    """Test ModelEvaluator functionality."""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Test model comparison
    results = ModelEvaluator.compare_models(
        X_train, X_test, y_train, y_test,
        models=['naive_bayes', 'svm']  # Test subset of models for speed
    )
    
    # Check results structure
    for model_type in ['naive_bayes', 'svm']:
        assert model_type in results
        model_results = results[model_type]
        assert 'accuracy' in model_results
        assert 'classification_report' in model_results
        assert 'class_metrics' in model_results
    
    # Test best model selection
    best_model, best_metrics = ModelEvaluator.get_best_model(results)
    assert best_model in ['naive_bayes', 'svm']
    assert best_metrics['accuracy'] == max(r['accuracy'] for r in results.values())
