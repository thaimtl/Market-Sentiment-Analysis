# Market-Sentiment-Analysis
This is my deliverable project for co-curricular course MAIS202: Introduction to Machine Learning by McGill 
## Features

- Load and preprocess financial sentiment data
- Fine-tune FinBERT for financial sentiment analysis
- Fetch and analyze real-time financial news headlines
- Visualize sentiment trends and distributions
- Web application for interactive analysis

## Project Structure

```
financial-sentiment-analysis/
│
├── data/                          # Data directory
│   ├── raw/                       # Original dataset files
│   └── processed/                 # Processed/cleaned data
│
├── models/                        # Saved models directory
│   └── finbert_finetuned/         # Fine-tuned FinBERT model 
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── src/                           # Source code
│   ├── __init__.py                # Makes src a Python package
│   ├── data/                      # Data processing scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py          # Data preprocessing functions (DONE)
│   │   └── fetch_news.py          # News fetching functions (DONE)
│   │
│   ├── models/                    # Model-related code
│   │   ├── __init__.py
│   │   ├── train.py               # Training functionality (DONE)
│   │   └── predict.py             # Prediction functionality
│   │
│   ├── visualization/             # Visualization scripts
│   │   ├── __init__.py
│   │   └── plots.py               # Plotting functions
│   │
│   └── app/                       # Web application code
│       ├── __init__.py
│       ├── main.py                # Main application entry point
│    
│
├── tests/                         # Unit tests
│   ├── test_preprocess.py
│   └── test_model.py
│
├── main.py                        # Main entry point
├── requirements.txt               # Project dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download and place the Kaggle Financial Sentiment Dataset in the `data/raw` directory as `data.csv`.
