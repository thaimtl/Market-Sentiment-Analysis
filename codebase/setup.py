from setuptools import setup, find_packages

setup(
    name="financial-sentiment-analysis",
    version="0.1.0",
    description="FinBERT-based financial sentiment analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "flask>=2.3.0",
        "nltk>=3.8.1",
        "requests>=2.31.0",
        "yfinance>=0.2.28",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pytest>=7.3.1",
    ],
    entry_points={
        'console_scripts': [
            'finbert-sentiment=main:main',
        ],
    },
)