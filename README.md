# Spam Email Detection Project

A comprehensive machine learning project for detecting spam emails using multiple classification models and an interactive dashboard for visualization and analysis.

## Features

### Multiple Classification Models
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- XGBoost
- LightGBM

### Text Preprocessing
- Email and URL removal
- Special character and number removal
- Tokenization
- Stop word removal
- Lemmatization
- TF-IDF vectorization

### Interactive Dashboard
- Model performance comparison
- Confusion matrices visualization
- ROC curves with AUC scores
- Real-time prediction interface
- Detailed metrics analysis

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curves
- Classification Reports

## Project Structure
```
spam-email-detection-1/
│
├── data/                 # Raw and preprocessed email data
├── src/                  # Source code
│   ├── preprocess.py    # Data preprocessing functions
│   ├── model.py         # Model training and evaluation
│   ├── predict.py       # Prediction functionality
│   ├── dashboard.py     # Interactive visualization dashboard
│   └── train_models.py  # Model training script
├── tests/               # Unit tests
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_predict.py
├── models/              # Saved models and preprocessor
├── Dockerfile          # Docker configuration
└── requirements.txt    # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd spam-email-detection-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

1. Prepare your dataset:
   - CSV file with two columns: "Messages" and "Category" (spam/ham)
   - Place it in the `data` directory

2. Train models using the training script:
```bash
python src/train_models.py --data path/to/your/emails.csv --test
```

This will:
- Preprocess your data
- Train all models
- Display performance metrics
- Save models to the `models` directory
- Run test predictions (if --test flag is used)

### Using the Dashboard

1. Start the dashboard:
```bash
python src/dashboard.py
```

2. Open your browser and go to `http://localhost:8050`

3. Dashboard features:
   - Upload your dataset
   - View model comparisons
   - Analyze confusion matrices
   - Compare ROC curves
   - Make real-time predictions
   - View detailed metrics

### Making Predictions Programmatically

```python
from src.predict import SpamPredictor

# Initialize predictor with saved models
predictor = SpamPredictor(
    model_path='models/spam_model_naive_bayes.joblib',
    preprocessor_path='models/preprocessor.joblib'
)

# Make predictions
message = "Hello, how are you?"
result = predictor.predict_message(message)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence_scores']['spam']:.2%}")
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Docker Support

1. Build the Docker image:
```bash
docker build -t spam-detection .
```

2. Run the container:
```bash
docker run -p 8050:8050 spam-detection
```

## Model Performance

The project evaluates multiple models and provides detailed performance metrics:
- Accuracy scores
- Precision and recall for spam detection
- F1-scores
- ROC curves with AUC scores
- Confusion matrices

View these metrics in the dashboard or during training.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dependencies

Key dependencies include:
- numpy==1.24.3
- pandas==2.0.2
- scikit-learn==1.2.2
- nltk==3.8.1
- dash==2.11.1
- dash-bootstrap-components==1.4.2
- plotly==5.15.0
- xgboost==2.0.1
- lightgbm==4.1.0

See `requirements.txt` for complete list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for text processing
- Scikit-learn for machine learning models
- Dash and Plotly for interactive visualization
- XGBoost and LightGBM teams for their excellent gradient boosting implementations
