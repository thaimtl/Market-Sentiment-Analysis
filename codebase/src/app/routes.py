from flask import Blueprint, render_template, request, jsonify
from src.app.services import AnalysisService, PredictorService
from src.app.validators import TextAnalysisValidator, StockAnalysisValidator
import traceback

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    """Home page with sentiment analysis interface"""
    return render_template('index.html')

@bp.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze sentiment of input text"""
    try:
        data = request.get_json()
        
        # Validate input
        validator = TextAnalysisValidator(data)
        if not validator.is_valid():
            return jsonify({'error': validator.get_error()}), 400
        
        # Check predictor availability
        if not PredictorService.is_ready():
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Analyze text
        result = AnalysisService.analyze_text(validator.text)
        
        return jsonify({
            'success': True,
            'text': validator.text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        print(f"Error in analyze_text: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/analyze_stock', methods=['POST'])
def analyze_stock():
    """Analyze sentiment of recent news for a stock ticker"""
    try:
        data = request.get_json()
        
        # Validate input
        validator = StockAnalysisValidator(data)
        if not validator.is_valid():
            return jsonify({'error': validator.get_error()}), 400
        
        # Check predictor availability
        if not PredictorService.is_ready():
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Analyze stock news
        result = AnalysisService.analyze_stock_news(
            validator.ticker, 
            validator.days, 
            validator.max_headlines
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_stock: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_ready': PredictorService.is_ready()
    })