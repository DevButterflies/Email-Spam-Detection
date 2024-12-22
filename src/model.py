from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import os
import numpy as np

class SpamDetectionModel:
    """Spam Detection Model with multiple classifier options."""
    
    AVAILABLE_MODELS = {
        'naive_bayes': MultinomialNB(),
        'svm': LinearSVC(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'lightgbm': LGBMClassifier(random_state=42)
    }
    
    def __init__(self, model_type='naive_bayes'):
        """Initialize the model with specified type."""
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model = self.AVAILABLE_MODELS[model_type]
    
    def train(self, X_train, y_train):
        """Train the model with the provided data."""
        # Convert sparse matrix to dense if needed (for XGBoost)
        if self.model_type in ['xgboost']:
            X_train = X_train.toarray()
            # Ensure no negative values in the input for MultinomialNB
        if isinstance(self.model, MultinomialNB):
            X_train = X_train.clip(min=0)  # Clip negative values to 0
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions on the input data."""
        # Convert sparse matrix to dense if needed (for XGBoost)
        if self.model_type in ['xgboost']:
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for predictions."""
        if not hasattr(self.model, 'predict_proba'):
            # For models like SVM that don't have predict_proba
            return None
        
        # Convert sparse matrix to dense if needed
        if self.model_type in ['xgboost']:
            X = X.toarray()
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return the performance metrics."""
        y_pred = self.predict(X_test)
        
        # Calculate precision, recall, and f1-score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'class_metrics': {
                'ham': {
                    'precision': precision[0],
                    'recall': recall[0],
                    'f1': f1[0]
                },
                'spam': {
                    'precision': precision[1],
                    'recall': recall[1],
                    'f1': f1[1]
                }
            }
        }
        
        # Add probability scores if available
        proba = self.predict_proba(X_test)
        if proba is not None:
            metrics['probability_scores'] = {
                'ham_proba': proba[:, 0],
                'spam_proba': proba[:, 1]
            }
        
        return metrics
    
    def save(self, model_path):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
    
    @classmethod
    def load(cls, model_path, model_type):
        """Load a trained model from disk."""
        model = cls(model_type)
        model.model = joblib.load(model_path)
        return model

class ModelEvaluator:
    """Class for comparing and evaluating multiple models."""
    
    @staticmethod
    def compare_models(X_train, X_test, y_train, y_test, models=None):
        """Compare different models and return their performance metrics."""
        if models is None:
            models = SpamDetectionModel.AVAILABLE_MODELS.keys()
        
        results = {}
        for model_type in models:
            print(f"\nTraining and evaluating {model_type}...")
            model = SpamDetectionModel(model_type)
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            # Store both metrics and trained model
            metrics['model_type'] = model_type
            metrics['trained_model'] = model
            results[model_type] = metrics
        
        return results
    
    @staticmethod
    def get_best_model(results, metric='accuracy'):
        """Get the best performing model based on the specified metric."""
        best_model = max(results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1]
