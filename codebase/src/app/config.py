import os

class Config:
    """Application configuration optimized for ML inference service"""
    
    # Model configuration
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'models', 'finbert_finetuned')
    
    # API configuration
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') or 'APRTD0XNPCP0J0YC'
    
    # Request timeout
    REQUEST_TIMEOUT = 30
    
    # Batch processing configuration
    BATCH_PROCESSING = {
        'MAX_BATCH_SIZE': 32,           # Maximum texts to process in one batch
        'OPTIMAL_BATCH_SIZE': 16,       # Optimal batch size for GPU utilization
        'MIN_BATCH_SIZE': 4,            # Minimum size to use batching
        'BATCH_TIMEOUT_MS': 100,        # Max wait time to accumulate batch (milliseconds)
        'ENABLE_AUTO_BATCHING': True,   # Automatically batch requests when beneficial
    }
    
    # Performance settings
    PERFORMANCE = {
        'ENABLE_MODEL_CACHING': True,
        'MAX_CACHE_SIZE': 1000,         # Maximum cached predictions
        'CACHE_TTL_SECONDS': 3600,      # Cache time-to-live
        'USE_HALF_PRECISION': False,    # Use FP16 for inference (GPU only)
    }