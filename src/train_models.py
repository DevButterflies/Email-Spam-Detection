import pandas as pd
import os
from preprocess import EmailPreprocessor
from model import SpamDetectionModel, ModelEvaluator
import joblib

def train_and_evaluate(data_path):
    """
    Train and evaluate all models on the dataset.
    
    Args:
        data_path: Path to the CSV file containing emails data
    """
    print("Step 1: Loading and preprocessing data...")
    preprocessor = EmailPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_path)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.joblib')
    print("Preprocessor saved to models/preprocessor.joblib")
    
    print("\nStep 2: Training and evaluating models...")
    results = ModelEvaluator.compare_models(X_train, X_test, y_train, y_test)
    
    # Print results and save models
    for model_name, metrics in results.items():
        print(f"\n{'-'*50}")
        print(f"Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Save the model
        model_path = f'models/spam_model_{model_name}.joblib'
        metrics['trained_model'].save(model_path)
        print(f"Model saved to {model_path}")
    
    # Find best model
    best_model, best_metrics = ModelEvaluator.get_best_model(results)
    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
    
    return results

def test_predictions(model_name='naive_bayes'):
    """
    Test predictions with a trained model.
    
    Args:
        model_name: Name of the model to use for predictions
    """
    from predict import SpamPredictor
    
    model_path = f'models/spam_model_{model_name}.joblib'
    preprocessor_path = 'models/preprocessor.joblib'
    
    predictor = SpamPredictor(model_path, preprocessor_path)
    
    # Test some example messages
    test_messages = [
        "Hello, how are you? Let's meet tomorrow.",
        "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize!",
        "Meeting scheduled for 2pm in the conference room.",
        "FREE VIAGRA! Best prices guaranteed! Click now!",
        "Hi John, I've reviewed your report and have some feedback."
    ]
    
    print("\nTesting predictions:")
    for message in test_messages:
        result = predictor.predict_message(message)
        print(f"\nMessage: {message}")
        print(f"Prediction: {result['prediction'].upper()}")
        if 'confidence_scores' in result:
            print(f"Confidence: {result['confidence_scores']['spam']:.2%} chance of being spam")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and test spam detection models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the CSV file containing emails data')
    parser.add_argument('--test', action='store_true',
                       help='Run test predictions after training')
    
    args = parser.parse_args()
    
    # Train models
    results = train_and_evaluate(args.data)
    
    # Test predictions if requested
    if args.test:
        print("\nTesting predictions with best model...")
        best_model = ModelEvaluator.get_best_model(results)[0]
        test_predictions(best_model)
