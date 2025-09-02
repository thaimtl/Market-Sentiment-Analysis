import os

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = True
    
    # Model configuration
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'models', 'finbert_finetuned')
    
    # API configuration
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    # Request timeout
    REQUEST_TIMEOUT = 30