import os
import sys
import traceback

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class PredictorService:
    """Service for managing the FinBERT predictor"""
    _predictor = None
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize the predictor service"""
        if cls._initialized:
            return True
            
        try:
            print("Initializing FinBERT predictor...")
            from src.models.predict import load_predictor
            cls._predictor = load_predictor()
            cls._initialized = True
            print("Predictor initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            traceback.print_exc()
            return False
    
    @classmethod
    def get_predictor(cls):
        """Get the predictor instance"""
        if not cls._initialized:
            if not cls.initialize():
                raise RuntimeError("Failed to initialize predictor")
        return cls._predictor
    
    @classmethod
    def is_ready(cls):
        """Check if predictor is ready"""
        return cls._initialized and cls._predictor is not None

class NewsService:
    """Service for fetching financial news"""
    
    @staticmethod
    def fetch_stock_news(ticker, days=7):
        """Fetch news for a stock ticker"""
        try:
            from src.data.fetch_news import fetch_stock_news
            return fetch_stock_news(ticker, days)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            raise

class AnalysisService:
    """Service for performing sentiment analysis"""
    
    @staticmethod
    def analyze_text(text):
        """Analyze sentiment of text"""
        predictor = PredictorService.get_predictor()
        return predictor.predict_sentiment(text, return_probabilities=True)
    
    @staticmethod
    def analyze_stock_news(ticker, days, max_headlines):
        """Analyze sentiment of stock news"""
        # Fetch news
        news_df = NewsService.fetch_stock_news(ticker, days)
        
        if news_df.empty:
            return {
                'success': True,
                'ticker': ticker,
                'message': f'No recent news found for {ticker}',
                'news_count': 0,
                'sentiment_summary': {},
                'news_items': []
            }
        
        # Analyze sentiment
        predictor = PredictorService.get_predictor()
        analyzed_df = predictor.analyze_dataframe(
            news_df, 
            text_column='Headline', 
            return_probabilities=True
        )
        
        # Get summary
        summary = predictor.get_sentiment_summary(analyzed_df)
        
        # Sort by date and limit results
        if 'Date' in analyzed_df.columns:
            analyzed_df = analyzed_df.sort_values('Date', ascending=False)
        limited_df = analyzed_df.head(max_headlines)
        
        # Prepare response
        news_items = []
        for _, row in limited_df.iterrows():
            news_items.append({
                'headline': row['Headline'],
                'date': row['Date'].strftime('%Y-%m-%d %H:%M:%S') if 'Date' in row else '',
                'sentiment': row['predicted_sentiment'],
                'confidence': round(row['confidence'], 3),
                'probabilities': {
                    'positive': round(row.get('prob_positive', 0), 3),
                    'neutral': round(row.get('prob_neutral', 0), 3),
                    'negative': round(row.get('prob_negative', 0), 3)
                }
            })
        
        return {
            'success': True,
            'ticker': ticker,
            'news_count': len(analyzed_df),
            'displayed_count': len(news_items),
            'sentiment_summary': summary,
            'news_items': news_items
        }