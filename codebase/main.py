#!/usr/bin/env python3
"""
Main entry point for the Financial Sentiment Analysis project
"""

import os
import sys
import argparse

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

def run_preprocessing():
    """Run data preprocessing"""
    print("Running data preprocessing...")
    try:
        from src.data.preprocess import load_and_preprocess_data
        df = load_and_preprocess_data()
        print(f"Successfully preprocessed {len(df)} samples")
        return True
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return False

def run_training():
    """Run model training"""
    print("Starting model training...")
    try:
        from src.models.train import FinancialSentimentTrainer, load_and_preprocess_data
        
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Initialize trainer
        trainer = FinancialSentimentTrainer()
        
        # Prepare data
        train_dataset, val_dataset = trainer.prepare_data(df)
        
        # Train model
        trainer.train(train_dataset, val_dataset)
        
        print("Training completed successfully!")
        return True
    except Exception as e:
        print(f"Error in training: {e}")
        return False

def run_tests():
    """Run unit tests"""
    import unittest
    
    print("Running unit tests...")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(current_dir, 'tests')
    
    if not os.path.exists(start_dir):
        print(f"❌ Tests directory not found at {start_dir}")
        print("Please create the tests directory and test files")
        return False
    
    suite = loader.discover(start_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False

def test_predictor():
    """Test the predictor with sample texts (legacy function)"""
    print("Testing optimized predictor...")
    try:
        from src.models.predict import load_predictor
        
        predictor = load_predictor()
        
        # Test with sample texts
        test_texts = [
            "Apple's stock price is soaring after excellent quarterly results!",
            "The company reported disappointing earnings and missed expectations.",
            "Trading volume was average today with mixed market signals.",
            "Investors are optimistic about the new product launch.",
            "Market volatility continues amid economic uncertainty."
        ]
        
        print("\nTesting Optimized FinBERT Predictor:")
        print("-" * 60)
        
        # Test batch processing
        import time
        start_time = time.time()
        batch_results = predictor.predict_sentiment_batch(test_texts, return_probabilities=True)
        batch_time = time.time() - start_time
        
        print(f"Batch processing time: {batch_time:.3f}s for {len(test_texts)} texts")
        print(f"Average time per text: {batch_time/len(test_texts):.3f}s")
        
        for i, (text, result) in enumerate(zip(test_texts, batch_results)):
            print(f"\n{i+1}. {text}")
            print(f"   Sentiment: {result['sentiment'].upper()}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
        # Performance metrics
        if hasattr(predictor, 'get_performance_metrics'):
            metrics = predictor.get_performance_metrics()
            if metrics:
                print("\nPerformance Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.3f}s")
        
        print("-" * 60)
        print("✅ Optimized predictor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing predictor: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_web_app():
    """Run the Flask web application"""
    print("Starting Flask web application...")
    try:
        # Change to app directory
        app_dir = os.path.join(current_dir, 'src', 'app')
        os.chdir(app_dir)
        
        from main import create_app
        app = create_app()
        
        print("=" * 60)
        print("🚀 FinBERT Sentiment Analysis App is running!")
        print("📊 Open your browser to: http://localhost:5000")
        print("⚡ Features: Optimized batch processing, caching, performance monitoring")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        return True
        
    except Exception as e:
        print(f"❌ Error running web app: {e}")
        import traceback
        traceback.print_exc()
        return False

def fetch_news_demo():
    """Demo the news fetching functionality"""
    print("Testing news fetching...")
    try:
        from src.data.fetch_news import fetch_stock_news
        
        ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
        if not ticker:
            ticker = "AAPL"
            
        days = 7
        print(f"Fetching {days} days of news for {ticker}...")
        
        news_df = fetch_stock_news(ticker, days=days)
        
        if not news_df.empty:
            print(f"\n✅ Successfully fetched {len(news_df)} news items for {ticker}")
            print("\nSample headlines:")
            for i, headline in enumerate(news_df['Headline'].head(3)):
                print(f"{i+1}. {headline}")
        else:
            print(f"❌ No news found for {ticker}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return False

def check_setup():
    """Check if the project is set up correctly"""
    print("Checking project setup...")
    
    # Check directory structure
    required_dirs = [
        'src',
        'src/data',
        'src/models', 
        'src/app',
        'data',
        'data/raw'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")
        return False
    
    # Check required files
    required_files = [
        'requirements.txt',
        'src/data/preprocess.py',
        'src/data/fetch_news.py',
        'src/models/train.py',
        'src/models/predict.py',
        'src/app/main.py',
        'src/app/config.py',
        'src/app/services.py',
        'src/app/routes.py',
        'src/app/utils.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Check for data file
    data_files = ['data/raw/data.csv', 'data.csv']
    data_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ Found data file: {data_file}")
            data_found = True
            break
    
    if not data_found:
        print("❌ No data file found. Expected data/raw/data.csv or data.csv")
        print("   Please download the dataset and place it in the correct location.")
    
    if not missing_dirs and not missing_files and data_found:
        print("✅ Project setup looks good!")
        print("✅ Optimized batch processing features available!")
        return True
    else:
        print("❌ Project setup has issues. Please fix the above problems.")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Financial Sentiment Analysis with FinBERT')
    parser.add_argument('command', choices=[
        'preprocess', 'train', 'test', 'webapp', 'demo', 'check'
    ], help='Command to run')
    
    # If no arguments provided, default to webapp
    if len(sys.argv) == 1:
        print("No command specified. Starting web application...")
        run_web_app()
        return
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 FinBERT Financial Sentiment Analysis")
    print("⚡ Optimized with Batch Processing & Performance Monitoring")
    print("=" * 60)
    
    if args.command == 'check':
        check_setup()
        
    elif args.command == 'preprocess':
        if not run_preprocessing():
            sys.exit(1)
            
    elif args.command == 'train':
        if not run_training():
            sys.exit(1)
            
    elif args.command == 'test':
        # Run comprehensive unit tests
        if not run_tests():
            print("\nFalling back to basic predictor test...")
            if not test_predictor():
                sys.exit(1)
            
    elif args.command == 'webapp':
        if not run_web_app():
            sys.exit(1)
            
    elif args.command == 'demo':
        if not fetch_news_demo():
            sys.exit(1)
    
    print("✅ Command completed successfully!")

if __name__ == "__main__":
    main()