from flask import Blueprint, render_template, request, jsonify
from src.app.services import AnalysisService, PredictorService
from src.app.validators import TextAnalysisValidator, StockAnalysisValidator, BatchAnalysisValidator
from src.app.utils import PerformanceMonitor
import traceback
from typing import List, Dict, Any

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    """Home page with sentiment analysis interface"""
    return render_template('index.html')

@bp.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze sentiment of input text"""
    performance_monitor = PerformanceMonitor()
    
    try:
        with performance_monitor.time_operation('request_processing'):
            data = request.get_json()
            
            # Validate input
            validator = TextAnalysisValidator(data)
            if not validator.is_valid():
                return jsonify({'error': validator.get_error()}), 400
            
            # Check predictor availability
            if not PredictorService.is_ready():
                return jsonify({'error': 'Predictor not initialized'}), 500
            
            # Analyze text
            with performance_monitor.time_operation('sentiment_analysis'):
                result = AnalysisService.analyze_text(validator.text)
            
            # Add performance metrics to response
            metrics = performance_monitor.get_metrics()
            
            return jsonify({
                'success': True,
                'text': validator.text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'processing_time_ms': round(metrics.get('request_processing', 0) * 1000, 2)
            })
        
    except Exception as e:
        print(f"Error in analyze_text: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/analyze_texts_batch', methods=['POST'])
def analyze_texts_batch():
    """Analyze sentiment of multiple texts using batch processing"""
    performance_monitor = PerformanceMonitor()
    
    try:
        with performance_monitor.time_operation('batch_request_processing'):
            data = request.get_json()
            
            # Validate input
            validator = BatchAnalysisValidator(data)
            if not validator.is_valid():
                return jsonify({'error': validator.get_error()}), 400
            
            # Check predictor availability
            if not PredictorService.is_ready():
                return jsonify({'error': 'Predictor not initialized'}), 500
            
            # Analyze texts using batch processing
            with performance_monitor.time_operation('batch_sentiment_analysis'):
                results = AnalysisService.analyze_texts_batch(validator.texts)
            
            # Calculate performance metrics
            metrics = performance_monitor.get_metrics()
            total_time = metrics.get('batch_request_processing', 0)
            analysis_time = metrics.get('batch_sentiment_analysis', 0)
            throughput = len(validator.texts) / analysis_time if analysis_time > 0 else 0
            
            return jsonify({
                'success': True,
                'results': results,
                'batch_size': len(validator.texts),
                'performance_metrics': {
                    'total_processing_time_ms': round(total_time * 1000, 2),
                    'analysis_time_ms': round(analysis_time * 1000, 2),
                    'throughput_texts_per_second': round(throughput, 2),
                    'average_time_per_text_ms': round((analysis_time / len(validator.texts)) * 1000, 2)
                }
            })
        
    except Exception as e:
        print(f"Error in analyze_texts_batch: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/analyze_stock', methods=['POST'])
def analyze_stock():
    """Analyze sentiment of recent news for a stock ticker using optimized batch processing"""
    performance_monitor = PerformanceMonitor()
    
    try:
        with performance_monitor.time_operation('stock_request_processing'):
            data = request.get_json()
            
            # Validate input
            validator = StockAnalysisValidator(data)
            if not validator.is_valid():
                return jsonify({'error': validator.get_error()}), 400
            
            # Check predictor availability
            if not PredictorService.is_ready():
                return jsonify({'error': 'Predictor not initialized'}), 500
            
            # Analyze stock news using optimized batch processing
            result = AnalysisService.analyze_stock_news(
                validator.ticker, 
                validator.days, 
                validator.max_headlines
            )
            
            # Add request-level performance metrics
            request_metrics = performance_monitor.get_metrics()
            if 'performance_metrics' in result:
                result['performance_metrics']['request_metrics'] = request_metrics
            
            return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_stock: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get comprehensive performance metrics and system status"""
    try:
        performance_summary = AnalysisService.get_performance_summary()
        return jsonify({
            'success': True,
            'performance_summary': performance_summary
        })
        
    except Exception as e:
        print(f"Error getting performance metrics: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear prediction cache for debugging/maintenance"""
    try:
        PredictorService.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/health')
def health_check():
    """Comprehensive health check endpoint with performance info"""
    try:
        performance_summary = AnalysisService.get_performance_summary()
        
        return jsonify({
            'status': 'healthy',
            'predictor_ready': PredictorService.is_ready(),
            'optimization_features': {
                'batch_processing': performance_summary['service_status']['batch_processing_enabled'],
                'caching': performance_summary['service_status']['cache_enabled'],
                'gpu_available': 'cuda' in str(PredictorService.get_predictor().device) if PredictorService.is_ready() else False
            },
            'performance_summary': performance_summary
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'predictor_ready': False
        }), 500