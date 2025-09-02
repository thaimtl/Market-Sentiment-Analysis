# Market-Sentiment-Analysis
This is my deliverable project for co-curricular course MAIS202: Introduction to Machine Learning by McGill 

## Features

- Load and preprocess financial sentiment data
- Fine-tune FinBERT for financial sentiment analysis
- Fetch and analyze real-time financial news headlines
- Visualize sentiment trends and distributions
- Web application for interactive analysis
- GPU acceleration support for training

## Project Back-end Structure

```
financial-sentiment-analysis/
│
├── data/                          # Data directory
│   ├── raw/                       # Original dataset files
│   │   └── data.csv              # Financial sentiment dataset (required)
│   └── processed/                 # Processed/cleaned data
│
├── models/                        # Saved models directory
│   └── finbert_finetuned/         # Fine-tuned FinBERT model (created after training)
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── src/                           # Source code
│   ├── __init__.py                # Makes src a Python package
│   ├── data/                      # Data processing scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py          # Data preprocessing functions
│   │   └── fetch_news.py          # News fetching functions
│   │
│   ├── models/                    # Model-related code
│   │   ├── __init__.py
│   │   ├── train.py               # Training functionality
│   │   └── predict.py             # Prediction functionality
│   │
│   ├── visualization/             # Visualization scripts
│   │   ├── __init__.py
│   │   └── plots.py               # Plotting functions
│   │
│   └── app/                       # Web application code
│       ├── __init__.py
│       └── main.py                # Flask web application
│    
├── tests/                         # Unit tests
│   ├── test_preprocess.py
│   └── test_model.py
│
├── main.py                        # Main entry point
├── setup.py                       # Package setup file
├── requirements.txt               # Project dependencies
```

## Installation
### TODO 
- Compare the fine-tuned model performance with the default one
- Fine tune it on a different dataset to minimize the neutral label
- Optimization and Improvement needed on fetching and sorting relevant headlines for a particular stock
  
### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/thaimtl/Market-Sentiment-Analysis.git
cd codebase
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install the Project
```bash
# Install project as editable package (recommended)
pip install -e .

# This automatically installs all dependencies from requirements.txt
```

### 4. Download Dataset
Download the [Kaggle Financial Sentiment Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis?resource=download) and place it in the `data/raw` directory as `data.csv`.

The dataset should contain columns: `Sentence` and `Sentiment`

### 5. GPU Setup (Optional but Recommended)
If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
# Check if you have CUDA
nvidia-smi

# Install CUDA-enabled PyTorch (adjust CUDA version as needed)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
## API Configuration

### Alpha Vantage API
The project uses Alpha Vantage for fetching real-time financial news.
- Get your free API key at: https://www.alphavantage.co/support/#api-key
- Update the API key in .env file

## Usage

### Quick Start (Using Pre-trained Model)
```bash
# Run the web application directly
python main.py webapp

# Open browser to: http://localhost:5000
```

### Complete Training Pipeline
```bash
# 1. Check project setup
python main.py check

# 2. Preprocess data
python main.py preprocess

# 3. Train custom model (15-60 minutes depending on hardware)
python main.py train

# 4. Test the trained model
python main.py test (TO BE COMPLETED ...)

# 5. Run web application with your trained model
cd src/app
python main.py 
```

### Command Line Options
```bash
python main.py [command]

Commands:
  check      - Verify project setup and requirements
  preprocess - Clean and prepare training data
  train      - Fine-tune FinBERT on your dataset
  test       - Test model predictions on sample texts
  webapp     - Start Flask web application
  demo       - Demo news fetching functionality
```

## Web Application Features

### Text Sentiment Analysis
- Paste any financial text or news headline
- Get sentiment prediction (Positive/Negative/Neutral)
- View confidence scores and probability distributions

### Stock News Analysis
- Enter stock ticker (e.g., AAPL, MSFT, TSLA)
- Fetch recent news headlines automatically
- Analyze sentiment for each news item
- View aggregate sentiment statistics

## Training Configuration

### Default Settings
- **Model**: ProsusAI/finbert (pre-trained financial BERT)
- **Epochs**: 4
- **Batch Size**: 24
- **Learning Rate**: 2e-5
- **Max Token Length**: 128

### Customization
Modify training parameters in `src/models/train.py`:
```python
# Adjust batch size based on your GPU memory
trainer.train(train_dataset, val_dataset, batch_size=16, num_epochs=3)
```

## Performance Expectations

### Training Time
- **GPU (RTX 4070)**: 5-15 minutes
- **CPU**: 30-60 minutes

### Model Accuracy
- **Pre-trained FinBERT**: ~85% accuracy on financial text
- **Fine-tuned model**: ~90%+ accuracy on similar data to training set

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall the project
pip install -e .
```

**CUDA Not Detected**
```bash
# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Memory Errors During Training**
```bash
# Reduce batch size in src/models/train.py
batch_size=8  # instead of 24
```

**API Rate Limits**
- Alpha Vantage free tier: 5 calls per minute
- Wait between requests or get a premium API key

### Logs and Debugging
- Training logs are saved to: `models/finbert_finetuned/logs/`
- Flask debug mode shows detailed error messages
- Use `python main.py check` to verify setup


## Acknowledgments

- [FinBERT](https://huggingface.co/ProsusAI/finbert) by ProsusAI
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [Alpha Vantage](https://www.alphavantage.co/) for financial news API
