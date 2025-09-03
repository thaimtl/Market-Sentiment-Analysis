import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Union, Optional, Any
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import clean_text
from src.app.utils import BatchProcessor, PerformanceMonitor, TextCache, RequestOptimizer
from src.app.config import Config

class OptimizedFinBERTPredictor:
    """Optimized FinBERT predictor with batch processing capabilities"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize optimized FinBERT predictor
        
        Args:
            model_path: Path to fine-tuned model. If None, uses pre-trained FinBERT
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.performance_monitor = PerformanceMonitor()
        self.cache = TextCache() if Config.PERFORMANCE['ENABLE_MODEL_CACHING'] else None
        
        # Try to load fine-tuned model first, fallback to pre-trained
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model_name = model_path
            self.is_finetuned = True
        else:
            print("Loading pre-trained FinBERT model...")
            self.model_name = "ProsusAI/finbert"
            self.is_finetuned = False
        
        self._load_model()
        self._setup_label_mappings()
    
    def _load_model(self):
        """Load model and tokenizer with optimizations"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Apply performance optimizations
            self.model.to(self.device)
            self.model.eval()
            
            # Enable half precision if configured and GPU available
            if (Config.PERFORMANCE['USE_HALF_PRECISION'] and 
                self.device.type == 'cuda'):
                self.model.half()
            
            # Skip model compilation for Windows compatibility
            # Triton compilation doesn't work reliably on Windows
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _setup_label_mappings(self):
        """Setup label mappings for sentiment interpretation"""
        if self.is_finetuned:
            # For our fine-tuned model: 0=positive, 1=negative, 2=neutral
            self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        else:
            # For pre-trained FinBERT: 0=positive, 1=negative, 2=neutral
            self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def predict_sentiment_batch(self, texts: List[str], 
                               return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts using optimized batch processing
        
        Args:
            texts: List of input texts to analyze
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        with self.performance_monitor.time_operation('batch_prediction'):
            # Clean and validate texts
            cleaned_texts = [clean_text(text) if isinstance(text, str) and text.strip() 
                           else "" for text in texts]
            
            # Check cache for existing predictions
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            if self.cache:
                for i, text in enumerate(cleaned_texts):
                    cached_result = self.cache.get(text)
                    if cached_result and not return_probabilities:
                        cached_results.append((i, cached_result))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = cleaned_texts
                uncached_indices = list(range(len(texts)))
            
            # Process uncached texts
            if uncached_texts:
                # Deduplicate for efficiency
                unique_texts, index_mapping = RequestOptimizer.deduplicate_texts(uncached_texts)
                
                # Process in optimal batches
                unique_results = self._process_text_batches(unique_texts, return_probabilities)
                
                # Reconstruct results
                uncached_results = RequestOptimizer.reconstruct_results(unique_results, index_mapping)
                
                # Cache new results
                if self.cache:
                    for text, result in zip(uncached_texts, uncached_results):
                        if text.strip():  # Only cache non-empty texts
                            self.cache.set(text, result)
            else:
                uncached_results = []
            
            # Combine cached and uncached results
            final_results = [None] * len(texts)
            
            # Insert cached results
            for idx, result in cached_results:
                final_results[idx] = result
            
            # Insert uncached results
            for i, idx in enumerate(uncached_indices):
                final_results[idx] = uncached_results[i]
            
            return final_results
    
    def _process_text_batches(self, texts: List[str], 
                             return_probabilities: bool) -> List[Dict[str, Any]]:
        """
        Process texts in optimized batches
        
        Args:
            texts: List of cleaned texts
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction results
        """
        if not texts:
            return []
        
        # Create optimal batches
        batches = BatchProcessor.create_optimal_batches(
            texts, Config.BATCH_PROCESSING['MAX_BATCH_SIZE'])
        
        all_results = []
        
        for batch in batches:
            batch_results = self._predict_batch(batch, return_probabilities)
            all_results.extend(batch_results)
        
        return all_results
    
    def _predict_batch(self, batch_texts: List[str], 
                      return_probabilities: bool) -> List[Dict[str, Any]]:
        """
        Process a single batch of texts
        
        Args:
            batch_texts: Batch of texts to process
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of predictions for the batch
        """
        if not batch_texts:
            return []
        
        # Handle empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(batch_texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            # Return neutral predictions for all empty texts
            return [self._get_neutral_prediction() for _ in batch_texts]
        
        # Tokenize batch
        with self.performance_monitor.time_operation('tokenization'):
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Predict
        with self.performance_monitor.time_operation('inference'):
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=-1).cpu().numpy()
                confidences = torch.max(predictions, dim=-1).values.cpu().numpy()
                probabilities = predictions.cpu().numpy()
        
        # Prepare results for valid texts
        valid_results = []
        for i, (pred_class, confidence, probs) in enumerate(
            zip(predicted_classes, confidences, probabilities)):
            
            sentiment = self.id2label[pred_class]
            result = {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "original_text": batch_texts[valid_indices[i]],
                "cleaned_text": valid_texts[i]
            }
            
            if return_probabilities:
                result["probabilities"] = {
                    self.id2label[j]: float(probs[j])
                    for j in range(len(self.id2label))
                }
            
            valid_results.append(result)
        
        # Reconstruct full results including empty texts
        full_results = []
        valid_idx = 0
        for i, text in enumerate(batch_texts):
            if i in valid_indices:
                full_results.append(valid_results[valid_idx])
                valid_idx += 1
            else:
                full_results.append(self._get_neutral_prediction())
        
        return full_results
    
    def _get_neutral_prediction(self) -> Dict[str, Any]:
        """Get neutral prediction for empty/invalid texts"""
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "original_text": "",
            "cleaned_text": ""
        }
    
    def predict_sentiment(self, text: str, 
                         return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Predict sentiment for a single text (backward compatibility)
        
        Args:
            text: Input text to analyze
            return_probabilities: Whether to return probability scores
            
        Returns:
            Sentiment prediction dictionary
        """
        results = self.predict_sentiment_batch([text], return_probabilities)
        return results[0] if results else self._get_neutral_prediction()
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = 'Headline',
                         return_probabilities: bool = False) -> pd.DataFrame:
        """
        Analyze sentiment for a pandas DataFrame using optimized batch processing
        
        Args:
            df: DataFrame containing text to analyze
            text_column: Column name containing text
            return_probabilities: Whether to include probability scores
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        print(f"Analyzing sentiment for {len(df)} texts using batch processing...")
        
        # Get predictions using batch processing
        texts = df[text_column].fillna("").tolist()
        
        with self.performance_monitor.time_operation('dataframe_analysis'):
            predictions = self.predict_sentiment_batch(texts, return_probabilities)
        
        # Add results to dataframe
        df['predicted_sentiment'] = [pred['sentiment'] for pred in predictions]
        df['confidence'] = [pred['confidence'] for pred in predictions]
        
        if return_probabilities:
            for sentiment in self.id2label.values():
                df[f'prob_{sentiment}'] = [
                    pred.get('probabilities', {}).get(sentiment, 0.0)
                    for pred in predictions
                ]
        
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of sentiment analysis
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Summary statistics dictionary
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
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance monitoring metrics"""
        return self.performance_monitor.get_metrics()
    
    def clear_cache(self):
        """Clear prediction cache"""
        if self.cache:
            self.cache.clear()


def load_predictor(model_path: Optional[str] = None) -> OptimizedFinBERTPredictor:
    """
    Convenience function to load the optimized predictor
    
    Args:
        model_path: Path to fine-tuned model
        
    Returns:
        Initialized optimized predictor
    """
    # Try to find fine-tuned model
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, "..", "..", "models", "finbert_finetuned")
        if os.path.exists(default_model_path):
            model_path = default_model_path
    
    return OptimizedFinBERTPredictor(model_path)


if __name__ == "__main__":
    # Simple test to verify the optimized predictor loads correctly
    print("Testing OptimizedFinBERTPredictor initialization...")
    try:
        predictor = load_predictor()
        print(f"✅ Predictor loaded successfully on {predictor.device}")
        
        # Quick functionality test
        result = predictor.predict_sentiment("Test sentiment analysis.", return_probabilities=True)
        print(f"✅ Basic prediction works: {result['sentiment']} ({result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Error during predictor test: {e}")
        import traceback
        traceback.print_exc()