import os
import sys
import traceback
from typing import List, Dict, Any, Optional

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.app.utils import PerformanceMonitor, BatchProcessor
from src.app.config import Config

class PredictorService:
    """Optimized service for managing the FinBERT predictor"""
    _predictor = None
    _initialized = False
    _performance_monitor = PerformanceMonitor()
    
    @classmethod
    def initialize(cls):
        """Initialize the predictor service with optimization"""
        if cls._initialized:
            return True
            
        try:
            print("Initializing optimized FinBERT predictor...")
            with cls._performance_monitor.time_operation('predictor_initialization'):
                from src.models.predict import load_predictor
                cls._predictor = load_predictor()
            
            cls._initialized = True
            print("Optimized predictor initialized successfully!")
            
            # Print initialization metrics
            metrics = cls._performance_monitor.get_metrics()
            if 'predictor_initialization' in metrics:
                print(f"Initialization time: {metrics['predictor_initialization']:.2f}s")
            
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
    
    @classmethod
    def get_performance_metrics(cls) -> Dict[str, float]:
        """Get service-level performance metrics"""
        service_metrics = cls._performance_monitor.get_metrics()
        
        # Add predictor metrics if available
        if cls._predictor and hasattr(cls._predictor, 'get_performance_metrics'):
            predictor_metrics = cls._predictor.get_performance_metrics()
            service_metrics.update({f"predictor_{k}": v for k, v in predictor_metrics.items()})
        
        return service_metrics
    
    @classmethod
    def clear_cache(cls):
        """Clear predictor cache"""
        if cls._predictor and hasattr(cls._predictor, 'clear_cache'):
            cls._predictor.clear_cache()

class NewsService:
    """Optimized service for fetching financial news"""
    
    @staticmethod
    def fetch_stock_news(ticker: str, days: int = 7):
        """
        Fetch news for a stock ticker with error handling
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            DataFrame containing news headlines
        """
        try:
            from src.data.fetch_news import fetch_stock_news
            return fetch_stock_news(ticker, days)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            raise

class AnalysisService:
    """Optimized service for performing sentiment analysis with batch processing"""
    
    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment analysis result
        """
        predictor = PredictorService.get_predictor()
        return predictor.predict_sentiment(text, return_probabilities=True)
    
    @staticmethod
    def analyze_texts_batch(texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts using optimized batch processing
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
        
        predictor = PredictorService.get_predictor()
        
        # Use batch processing if beneficial
        if BatchProcessor.should_use_batching(len(texts)):
            print(f"Using batch processing for {len(texts)} texts")
            return predictor.predict_sentiment_batch(texts, return_probabilities=True)
        else:
            # Process individually for small numbers
            print(f"Processing {len(texts)} texts individually")
            return [predictor.predict_sentiment(text, return_probabilities=True) 
                   for text in texts]
    
    @staticmethod
    def analyze_stock_news(ticker: str, days: int, max_headlines: int) -> Dict[str, Any]:
        """
        Analyze sentiment of stock news using optimized batch processing
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            max_headlines: Maximum number of headlines to analyze
            
        Returns:
            Complete analysis results with batch processing metrics
        """
        performance_monitor = PerformanceMonitor()
        
        # Fetch news
        with performance_monitor.time_operation('news_fetching'):
            news_df = NewsService.fetch_stock_news(ticker, days)
        
        if news_df.empty:
            return {
                'success': True,
                'ticker': ticker,
                'message': f'No recent news found for {ticker}',
                'news_count': 0,
                'sentiment_summary': {},
                'news_items': [],
                'performance_metrics': performance_monitor.get_metrics()
            }
        
        # Prepare headlines for batch processing
        headlines = news_df['Headline'].fillna("").tolist()
        
        # Analyze sentiment using optimized batch processing
        with performance_monitor.time_operation('sentiment_analysis'):
            predictor = PredictorService.get_predictor()
            
            # Use the optimized batch processing from the predictor
            analyzed_df = predictor.analyze_dataframe(
                news_df, 
                text_column='Headline', 
                return_probabilities=True
            )
        
        # Get summary
        with performance_monitor.time_operation('summary_generation'):
            summary = predictor.get_sentiment_summary(analyzed_df)
        
        # Sort by date and limit results
        with performance_monitor.time_operation('result_preparation'):
            if 'Date' in analyzed_df.columns:
                analyzed_df = analyzed_df.sort_values('Date', ascending=False)
            limited_df = analyzed_df.head(max_headlines)
            
            # Prepare response with optimized data structure
            news_items = AnalysisService._prepare_news_items(limited_df)
        
        # Collect all performance metrics
        analysis_metrics = performance_monitor.get_metrics()
        predictor_metrics = predictor.get_performance_metrics()
        service_metrics = PredictorService.get_performance_metrics()
        
        # Calculate efficiency metrics
        total_headlines = len(headlines)
        analysis_time = analysis_metrics.get('sentiment_analysis', 0)
        throughput = total_headlines / analysis_time if analysis_time > 0 else 0
        
        return {
            'success': True,
            'ticker': ticker,
            'news_count': len(analyzed_df),
            'displayed_count': len(news_items),
            'sentiment_summary': summary,
            'news_items': news_items,
            'performance_metrics': {
                'analysis_metrics': analysis_metrics,
                'predictor_metrics': predictor_metrics,
                'service_metrics': service_metrics,
                'efficiency': {
                    'total_headlines': total_headlines,
                    'analysis_time_seconds': analysis_time,
                    'throughput_headlines_per_second': round(throughput, 2),
                    'used_batch_processing': BatchProcessor.should_use_batching(total_headlines)
                }
            }
        }
    
    @staticmethod
    def _prepare_news_items(analyzed_df) -> List[Dict[str, Any]]:
        """
        Prepare news items for API response with optimized data structure
        
        Args:
            analyzed_df: DataFrame with analysis results
            
        Returns:
            List of formatted news items
        """
        news_items = []
        
        for _, row in analyzed_df.iterrows():
            item = {
                'headline': row['Headline'],
                'date': row['Date'].strftime('%Y-%m-%d %H:%M:%S') if 'Date' in row else '',
                'sentiment': row['predicted_sentiment'],
                'confidence': round(row['confidence'], 3)
            }
            
            # Add probabilities if available
            prob_columns = [col for col in row.index if col.startswith('prob_')]
            if prob_columns:
                item['probabilities'] = {
                    col.replace('prob_', ''): round(row[col], 3)
                    for col in prob_columns
                }
            
            news_items.append(item)
        
        return news_items
    
    @staticmethod
    def get_performance_summary() -> Dict[str, Any]:
        """
        Get comprehensive performance summary for monitoring
        
        Returns:
            Performance summary with key metrics
        """
        service_metrics = PredictorService.get_performance_metrics()
        
        return {
            'service_status': {
                'predictor_ready': PredictorService.is_ready(),
                'cache_enabled': Config.PERFORMANCE['ENABLE_MODEL_CACHING'],
                'batch_processing_enabled': Config.BATCH_PROCESSING['ENABLE_AUTO_BATCHING']
            },
            'performance_metrics': service_metrics,
            'configuration': {
                'max_batch_size': Config.BATCH_PROCESSING['MAX_BATCH_SIZE'],
                'optimal_batch_size': Config.BATCH_PROCESSING['OPTIMAL_BATCH_SIZE'],
                'min_batch_size': Config.BATCH_PROCESSING['MIN_BATCH_SIZE']
            }
        }