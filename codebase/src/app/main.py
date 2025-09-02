from flask import Flask
from src.app.config import Config
from src.app.routes import bp as main_bp
from src.app.services import PredictorService

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Load configuration
    app.config.from_object(Config)
    
    # Initialize services
    PredictorService.initialize()
    
    # Register blueprints
    app.register_blueprint(main_bp)
    
    return app

def main():
    """Main entry point"""
    app = create_app()
    
    print("Starting FinBERT Sentiment Analysis App...")
    print("Open your browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()