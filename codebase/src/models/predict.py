import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import clean_text

class FinBERTPredictor:
    def __init__(self, model_path=None):
        """
        Initialize FinBERT predictor
        
        Parameters:
        -----------
        model_path : str, optional
            Path to fine-tuned model. If None, uses pre-trained FinBERT
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load fine-tuned model first, fallback to pre-trained
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model_name = model_path
            self.is_finetuned = True
        else:
            print("Loading pre-trained FinBERT model...")
            self.model_name = "ProsusAI/finbert"
            self.is_finetuned = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Label mappings for FinBERT
            if self.is_finetuned:
                # For our fine-tuned model: 0=positive, 1=negative, 2=neutral
                self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
            else:
                # For pre-trained FinBERT: 0=positive, 1=negative, 2=neutral
                self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
            
            self.label2id = {v: k for k, v in self.id2label.items()}
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_sentiment(self, text, return_probabilities=False):
        """
        Predict sentiment for a single text
        
        Parameters:
        -----------
        text : str
            Input text to analyze
        return_probabilities : bool, default=False
            Whether to return probability scores
            
        Returns:
        --------
        dict or str
            Sentiment prediction and optionally probabilities
        """
        if not isinstance(text, str) or not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).cpu().numpy()[0]
            confidence = float(torch.max(predictions).cpu().numpy())
        
        sentiment = self.id2label[predicted_class]
        
        if return_probabilities:
            probabilities = {
                self.id2label[i]: float(predictions[0][i].cpu().numpy())
                for i in range(len(self.id2label))
            }
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": probabilities,
                "original_text": text,
                "cleaned_text": cleaned_text
            }
        else:
            return {
                "sentiment": sentiment,
                "confidence": confidence
            }
    
    def predict_batch(self, texts, return_probabilities=False, batch_size=32):
        """
        Predict sentiment for multiple texts
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze
        return_probabilities : bool, default=False
            Whether to return probability scores
        batch_size : int, default=32
            Batch size for processing
            
        Returns:
        --------
        list
            List of predictions
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict_sentiment(text, return_probabilities)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def analyze_dataframe(self, df, text_column='Headline', return_probabilities=False):
        """
        Analyze sentiment for a pandas DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing text to analyze
        text_column : str, default='Headline'
            Column name containing text
        return_probabilities : bool, default=False
            Whether to include probability scores
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment analysis results
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        print(f"Analyzing sentiment for {len(df)} texts...")
        
        # Get predictions
        texts = df[text_column].fillna("").tolist()
        predictions = self.predict_batch(texts, return_probabilities)
        
        # Add results to dataframe
        df['predicted_sentiment'] = [pred['sentiment'] for pred in predictions]
        df['confidence'] = [pred['confidence'] for pred in predictions]
        
        if return_probabilities:
            for sentiment in self.id2label.values():
                df[f'prob_{sentiment}'] = [
                    pred['probabilities'][sentiment] if 'probabilities' in pred else 0.0
                    for pred in predictions
                ]
        
        return df
    
    def get_sentiment_summary(self, df):
        """
        Get summary statistics of sentiment analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sentiment analysis results
            
        Returns:
        --------
        dict
            Summary statistics
        """
        if 'predicted_sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['predicted_sentiment'].value_counts()
        total = len(df)
        
        summary = {
            'total_texts': total,
            'sentiment_distribution': {
                sentiment: {
                    'count': int(count),
                    'percentage': round((count / total) * 100, 2)
                }
                for sentiment, count in sentiment_counts.items()
            },
            'average_confidence': round(df['confidence'].mean(), 3),
            'most_common_sentiment': sentiment_counts.index[0] if not sentiment_counts.empty else 'neutral'
        }
        
        return summary


def load_predictor(model_path=None):
    """
    Convenience function to load the predictor
    
    Parameters:
    -----------
    model_path : str, optional
        Path to fine-tuned model
        
    Returns:
    --------
    FinBERTPredictor
        Initialized predictor
    """
    # Try to find fine-tuned model
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, "..", "..", "models", "finbert_finetuned")
        if os.path.exists(default_model_path):
            model_path = default_model_path
    
    return FinBERTPredictor(model_path)


if __name__ == "__main__":
    # Test the predictor
    predictor = load_predictor()
    
    # Test with sample texts
    test_texts = [
        "Apple's stock price is soaring after excellent quarterly results!",
        "The company reported disappointing earnings and missed expectations.",
        "Trading volume was average today with mixed market signals."
    ]
    
    print("Testing FinBERT Predictor:")
    print("-" * 50)
    
    for text in test_texts:
        result = predictor.predict_sentiment(text, return_probabilities=True)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Probabilities:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.3f}")
        print("-" * 50)