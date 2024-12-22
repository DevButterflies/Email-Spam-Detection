import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

class EmailPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with necessary NLTK downloads."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
    
    def clean_text(self, text):
        """Clean and preprocess the text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def prepare_data(self, data, test_size=0.2, random_state=42):
        """
        Load, clean, and prepare the data for training.
        
        Args:
            data: Either a pandas DataFrame or path to CSV file
            test_size: Proportion of dataset to include in the test split
            random_state: Random state for reproducibility
            
        Returns:
            X_train_vectorized, X_test_vectorized, y_train, y_test
        """
        # Load the dataset if path is provided
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Data file not found: {data}")
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        # Validate DataFrame structure
        required_columns = {'Messages', 'Category'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Clean the messages
        print("Cleaning messages...")
        df['cleaned_messages'] = df['Messages'].apply(self.clean_text)
        
        # Convert categories to binary values
        print("Converting categories to binary...")
        df['Category'] = (df['Category'].str.lower() == 'spam').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_messages'],
            df['Category'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['Category']
        )
        
        # Check if X_train is empty
        if X_train.empty:
            raise ValueError("X_train is empty. Please check the data.")
    
        # Vectorize the text data
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    def transform_text(self, text):
        """Transform new text data for prediction."""
        cleaned_text = self.clean_text(text)
        return self.vectorizer.transform([cleaned_text])
    
    def save(self, filepath):
        """Save the preprocessor to disk."""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a preprocessor from disk."""
        return joblib.load(filepath)
