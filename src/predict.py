from preprocess import EmailPreprocessor
from model import SpamDetectionModel
import joblib
import os

class SpamPredictor:
    """Class for making predictions using trained spam detection models."""
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with paths to saved model and preprocessor.
        
        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        # Extract model type from filename
        model_type = os.path.basename(model_path).split('_')[-1].split('.')[0]
        self.model = SpamDetectionModel.load(model_path, model_type)
        self.preprocessor = EmailPreprocessor.load(preprocessor_path)
    
    def predict_message(self, message):
        """
        Predict whether a message is spam or ham.
        
        Args:
            message: Text message to classify
            
        Returns:
            dict containing prediction and confidence scores
        """
        # Preprocess the message
        X = self.preprocessor.transform_text(message)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'is_spam': bool(prediction)
        }
        
        # Add probability scores if available
        proba = self.model.predict_proba(X)
        if proba is not None:
            result['confidence_scores'] = {
                'ham': float(proba[0][0]),
                'spam': float(proba[0][1])
            }
        
        return result
    
    def predict_batch(self, messages):
        """
        Predict for a batch of messages.
        
        Args:
            messages: List of text messages to classify
            
        Returns:
            List of prediction results
        """
        return [self.predict_message(message) for message in messages]

def load_predictor(model_dir='models'):
    """
    Load the predictor with the latest trained model.
    
    Args:
        model_dir: Directory containing the model and preprocessor files
        
    Returns:
        SpamPredictor instance
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find the best performing model (assuming the filename contains the accuracy)
    model_files = [f for f in os.listdir(model_dir) if f.startswith('spam_model_')]
    if not model_files:
        raise FileNotFoundError("No model files found. Please train the models first.")
    
    model_path = os.path.join(model_dir, model_files[0])
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    
    return SpamPredictor(model_path, preprocessor_path)
