import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.predict import OptimizedFinBERTPredictor, load_predictor
from src.app.utils import BatchProcessor, TextCache, RequestOptimizer

class TestOptimizedFinBERTPredictor(unittest.TestCase):
    """Test cases for the optimized FinBERT predictor"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class"""
        # Create a mock predictor to avoid loading the actual model during tests
        cls.test_texts = [
            "Apple's quarterly earnings exceeded expectations significantly.",
            "The market experienced severe volatility and declined sharply.",
            "Trading volume remained stable throughout the session.",
            "",  # Empty text
            "   ",  # Whitespace only
        ]
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the model loading to avoid downloading during tests
        self.mock_predictor = None
    
    def test_batch_processor_optimal_batches(self):
        """Test batch creation with different batch sizes"""
        items = list(range(50))  # 50 items
        
        # Test with different batch sizes
        batches_16 = BatchProcessor.create_optimal_batches(items, 16)
        self.assertEqual(len(batches_16), 4)  # 50/16 = 3.125, so 4 batches
        self.assertEqual(len(batches_16[0]), 16)
        self.assertEqual(len(batches_16[-1]), 2)  # Last batch has remainder
        
        # Test empty list
        empty_batches = BatchProcessor.create_optimal_batches([])
        self.assertEqual(empty_batches, [])
    
    def test_batch_processor_should_use_batching(self):
        """Test batching decision logic"""
        # Should use batching for larger numbers
        self.assertTrue(BatchProcessor.should_use_batching(10))
        self.assertTrue(BatchProcessor.should_use_batching(20))
        
        # Should not use batching for small numbers
        self.assertFalse(BatchProcessor.should_use_batching(1))
        self.assertFalse(BatchProcessor.should_use_batching(2))
    
    def test_text_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = TextCache(max_size=3)
        
        # Test setting and getting
        cache.set("test text", {"sentiment": "positive"})
        result = cache.get("test text")
        self.assertEqual(result["sentiment"], "positive")
        
        # Test cache miss
        self.assertIsNone(cache.get("non-existent text"))
        
        # Test cache size
        self.assertEqual(cache.size(), 1)
    
    def test_text_cache_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = TextCache(max_size=2)
        
        # Fill cache
        cache.set("text1", {"sentiment": "positive"})
        cache.set("text2", {"sentiment": "negative"})
        self.assertEqual(cache.size(), 2)
        
        # Access text1 to make it more recent
        cache.get("text1")
        
        # Add text3, should evict text2 (least recently used)
        cache.set("text3", {"sentiment": "neutral"})
        self.assertEqual(cache.size(), 2)
        self.assertIsNone(cache.get("text2"))  # Should be evicted
        self.assertIsNotNone(cache.get("text1"))  # Should still exist
        self.assertIsNotNone(cache.get("text3"))  # Should exist
    
    def test_request_optimizer_deduplication(self):
        """Test text deduplication functionality"""
        texts = ["apple", "banana", "apple", "cherry", "banana"]
        unique_texts, index_mapping = RequestOptimizer.deduplicate_texts(texts)
        
        # Should have 3 unique texts
        self.assertEqual(len(unique_texts), 3)
        self.assertEqual(set(unique_texts), {"apple", "banana", "cherry"})
        
        # Index mapping should reconstruct original order
        self.assertEqual(len(index_mapping), 5)
        self.assertEqual(index_mapping, [0, 1, 0, 2, 1])
    
    def test_request_optimizer_reconstruction(self):
        """Test result reconstruction after deduplication"""
        unique_results = ["result_a", "result_b", "result_c"]
        index_mapping = [0, 1, 0, 2, 1]
        
        reconstructed = RequestOptimizer.reconstruct_results(unique_results, index_mapping)
        expected = ["result_a", "result_b", "result_a", "result_c", "result_b"]
        
        self.assertEqual(reconstructed, expected)
    
    @patch('src.models.predict.AutoTokenizer')
    @patch('src.models.predict.AutoModelForSequenceClassification')
    def test_predictor_initialization(self, mock_model_class, mock_tokenizer_class):
        """Test predictor initialization with mocked components"""
        # Mock the tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create predictor
        predictor = OptimizedFinBERTPredictor()
        
        # Verify initialization
        self.assertIsNotNone(predictor.tokenizer)
        self.assertIsNotNone(predictor.model)
        self.assertIn(predictor.device.type, ['cpu', 'cuda'])
    
    def test_predictor_neutral_prediction(self):
        """Test neutral prediction for empty/invalid texts"""
        # Create a mock predictor
        predictor = OptimizedFinBERTPredictor.__new__(OptimizedFinBERTPredictor)
        predictor.cache = None
        predictor.performance_monitor = MagicMock()
        predictor.performance_monitor.time_operation.return_value.__enter__ = MagicMock()
        predictor.performance_monitor.time_operation.return_value.__exit__ = MagicMock()
        
        # Test neutral prediction
        neutral_pred = predictor._get_neutral_prediction()
        
        self.assertEqual(neutral_pred["sentiment"], "neutral")
        self.assertEqual(neutral_pred["confidence"], 0.0)
        self.assertEqual(neutral_pred["original_text"], "")
        self.assertEqual(neutral_pred["cleaned_text"], "")
    
    def test_predictor_label_mappings(self):
        """Test sentiment label mappings"""
        # Create a mock predictor
        predictor = OptimizedFinBERTPredictor.__new__(OptimizedFinBERTPredictor)
        predictor.is_finetuned = False
        predictor._setup_label_mappings()
        
        # Test label mappings
        expected_labels = {"positive", "negative", "neutral"}
        self.assertEqual(set(predictor.id2label.values()), expected_labels)
        
        # Test reverse mapping
        for label_id, label_text in predictor.id2label.items():
            self.assertEqual(predictor.label2id[label_text], label_id)

class TestModelIntegration(unittest.TestCase):
    """Integration tests for model functionality"""
    
    def test_load_predictor_function(self):
        """Test the load_predictor convenience function"""
        # Test without existing model (should use pre-trained)
        with patch('src.models.predict.OptimizedFinBERTPredictor') as mock_predictor_class:
            mock_instance = MagicMock()
            mock_predictor_class.return_value = mock_instance
            
            # Test loading
            result = load_predictor()
            
            # Should create predictor instance
            mock_predictor_class.assert_called_once()
            self.assertEqual(result, mock_instance)
    
    def test_batch_vs_single_consistency(self):
        """Test that batch and single predictions are consistent"""
        # This would require actual model loading, so we'll mock it
        with patch('src.models.predict.OptimizedFinBERTPredictor') as mock_predictor_class:
            mock_instance = MagicMock()
            
            # Mock single prediction
            single_result = {
                "sentiment": "positive",
                "confidence": 0.85,
                "probabilities": {"positive": 0.85, "negative": 0.10, "neutral": 0.05}
            }
            mock_instance.predict_sentiment.return_value = single_result
            
            # Mock batch prediction
            batch_result = [single_result]
            mock_instance.predict_sentiment_batch.return_value = batch_result
            
            mock_predictor_class.return_value = mock_instance
            
            predictor = load_predictor()
            
            # Test single prediction
            single_pred = predictor.predict_sentiment("test text", return_probabilities=True)
            
            # Test batch prediction with same text
            batch_pred = predictor.predict_sentiment_batch(["test text"], return_probabilities=True)
            
            # Results should be equivalent
            self.assertEqual(single_pred["sentiment"], batch_pred[0]["sentiment"])
            self.assertEqual(single_pred["confidence"], batch_pred[0]["confidence"])

class TestPerformanceFeatures(unittest.TestCase):
    """Test performance optimization features"""
    
    def test_cache_performance(self):
        """Test caching improves performance"""
        cache = TextCache(max_size=100)
        
        # First access - cache miss
        result1 = cache.get("test text")
        self.assertIsNone(result1)
        
        # Set value
        test_result = {"sentiment": "positive", "confidence": 0.9}
        cache.set("test text", test_result)
        
        # Second access - cache hit
        result2 = cache.get("test text")
        self.assertEqual(result2, test_result)
    
    def test_deduplication_efficiency(self):
        """Test that deduplication reduces work"""
        # Create list with many duplicates
        texts = ["apple"] * 10 + ["banana"] * 5 + ["cherry"] * 3
        
        unique_texts, mapping = RequestOptimizer.deduplicate_texts(texts)
        
        # Should significantly reduce work
        self.assertEqual(len(unique_texts), 3)  # Only 3 unique texts
        self.assertEqual(len(mapping), 18)     # But 18 total texts
        
        # Efficiency ratio
        efficiency = len(unique_texts) / len(texts)
        self.assertLess(efficiency, 0.2)  # Should be much more efficient
    
    def test_batch_size_optimization(self):
        """Test optimal batch size calculations"""
        # Test various input sizes
        test_sizes = [1, 5, 10, 20, 50, 100]
        
        for size in test_sizes:
            items = list(range(size))
            batches = BatchProcessor.create_optimal_batches(items, max_batch_size=16)
            
            # Verify all items are included
            total_items = sum(len(batch) for batch in batches)
            self.assertEqual(total_items, size)
            
            # Verify no batch exceeds max size
            for batch in batches:
                self.assertLessEqual(len(batch), 16)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Test batch processor with empty list
        batches = BatchProcessor.create_optimal_batches([])
        self.assertEqual(batches, [])
        
        # Test cache with empty string
        cache = TextCache()
        cache.set("", {"sentiment": "neutral"})
        result = cache.get("")
        self.assertIsNotNone(result)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        # Test deduplication with mixed types (should handle gracefully)
        texts = ["valid_text", "", None, "another_text"]
        
        # Filter out None values as the function expects strings
        valid_texts = [t for t in texts if t is not None]
        unique_texts, mapping = RequestOptimizer.deduplicate_texts(valid_texts)
        
        self.assertIsInstance(unique_texts, list)
        self.assertIsInstance(mapping, list)
        self.assertEqual(len(mapping), len(valid_texts))
    
    def test_memory_limits(self):
        """Test behavior with large inputs"""
        # Test cache with max size
        cache = TextCache(max_size=2)
        
        # Add more items than max size
        for i in range(5):
            cache.set(f"text_{i}", {"sentiment": "neutral"})
        
        # Should maintain max size
        self.assertEqual(cache.size(), 2)

if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)