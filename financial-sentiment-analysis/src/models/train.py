import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from datasets import Dataset
import os
import sys
import pandas as pd

# Add parent directory to path to import from src.data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import load_and_preprocess_data

def compute_metrics(pred):
    """
    Compute evaluation metrics for the model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class FinancialSentimentTrainer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize the trainer
        
        Parameters:
        -----------
        model_name : str, default="ProsusAI/finbert"
            Name of the pre-trained model
        output_dir : str, default="../../models/finbert_finetuned"
            Directory to save fine-tuned model
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        self.output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "models", "finbert_finetuned"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map dataset labels to model labels (FinBERT: 0=positive, 1=negative, 2=neutral)
        # Our numeric dataset: 1=positive, -1=negative, 0=neutral
        self.numeric_label_map = {1: 0, -1: 1, 0: 2}
        
        # Map for text labels to numeric labels
        self.text_to_numeric = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        # Map for FinBERT labels (text or numeric directly to model labels)
        self.text_label_map = {
            'positive': 0,
            'neutral': 2,
            'negative': 1
        }
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
    
    def prepare_data(self, df, test_size=0.2, max_length=300): #MAX LENGTH IS NUMBER OF TOKENS IN 1 SENTENCE. <= 315
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with 'Sentence' and 'Sentiment' columns
        test_size : float, default=0.2
            Fraction of data to use for validation
        max_length : int, default=128
            Maximum token length
            
        Returns:
        --------
        tuple
            (train_dataset, val_dataset)
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Get information about the DataFrame
        print("\nDataFrame information:")
        print(df.info())
        
        # Display the first few rows
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for NaN values
        if df['Sentiment'].isna().any():
            print(f"Warning: Found {df['Sentiment'].isna().sum()} NaN values in Sentiment column. Dropping these rows.")
            df = df.dropna(subset=['Sentiment'])
        
        # Check if the sentiment column contains text or numeric values
        sentiment_type = 'unknown'
        if df['Sentiment'].dtype == 'object':
            # Check if values are textual sentiments
            sample_values = df['Sentiment'].dropna().unique()
            print(f"\nUnique sentiment values: {sample_values}")
            
            if any(isinstance(val, str) for val in sample_values):
                sentiment_type = 'text'
                
                # Convert text sentiment values to numeric
                print("Converting text sentiment values to numeric...")
                
                # Create a mapping function that handles case and common variations
                def map_sentiment(val):
                    if not isinstance(val, str):
                        return val
                    
                    val = val.lower().strip()
                    if val in ['positive', 'pos', '1', 'p']:
                        return 1
                    elif val in ['negative', 'neg', '-1', 'n']:
                        return -1
                    elif val in ['neutral', 'neu', '0']:
                        return 0
                    else:
                        return None
                
                df['numeric_sentiment'] = df['Sentiment'].apply(map_sentiment)
                
                # Check for any unmapped values
                if df['numeric_sentiment'].isna().any():
                    unmapped = df[df['numeric_sentiment'].isna()]['Sentiment'].unique()
                    print(f"Warning: Could not map these sentiment values: {unmapped}")
                    print("Dropping rows with unmapped sentiment values.")
                    df = df.dropna(subset=['numeric_sentiment'])
                
                # Use the numeric sentiment for further processing
                df['Sentiment'] = df['numeric_sentiment']
                
        # Ensure Sentiment is numeric at this point
        df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')
        
        # Check for NaN values again after conversion
        if df['Sentiment'].isna().any():
            print(f"Warning: After conversion, found {df['Sentiment'].isna().sum()} NaN values in Sentiment column. Dropping these rows.")
            df = df.dropna(subset=['Sentiment'])
        
        # Filter for valid sentiment values
        valid_sentiments = [1, 0, -1]  # Positive, Neutral, Negative
        mask = df['Sentiment'].isin(valid_sentiments)
        if not mask.all():
            print(f"Warning: Found {(~mask).sum()} rows with invalid sentiment values. Dropping these rows.")
            df = df[mask]
        
        # Map labels for FinBERT
        df['model_label'] = df['Sentiment'].map(self.numeric_label_map)
        
        # Verify no NaN values in model_label
        if df['model_label'].isna().any():
            missing_labels = df[df['model_label'].isna()]['Sentiment'].unique()
            print(f"Error: Found unlabeled sentiment values: {missing_labels}")
            print("Please check your label mapping. Dropping these rows for now.")
            df = df.dropna(subset=['model_label'])
        
        # Split data
        print(f"\nSplitting data with {test_size*100}% for validation...")
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df['model_label'], random_state=42)
        
        print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
        
        # Display sentiment distribution after preprocessing
        print("\nSentiment distribution in training data:")
        train_dist = train_df['Sentiment'].value_counts()
        total = len(train_df)
        for sentiment, count in train_dist.items():
            print(f"  {sentiment}: {count} ({count/total*100:.1f}%)")
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Determine the text column name
        text_column = 'processed_text' if 'processed_text' in train_df.columns else 'Sentence'
        
        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column], 
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Apply tokenization
        print(f"\nTokenizing data using {text_column} column...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset = train_dataset.rename_column('model_label', 'labels')
        val_dataset = val_dataset.rename_column('model_label', 'labels')
        
        train_dataset.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, batch_size=20, num_epochs=4):
        """
        Train the model
        
        Parameters:
        -----------
        train_dataset : datasets.Dataset
            Training dataset
        val_dataset : datasets.Dataset
            Validation dataset
        batch_size : int, default=16. ## GOOD RANGE IS 16 -24, INCREASE IT IF TRAINING IS TOO SLOW
            Batch size for training
        num_epochs : int, default=3.  ## TO SEE, IN GENRAL, MORE EPOCHS BETTER LEARNING OF PATTERNS, RISK OVERFITTING. TOO FEW, RISK UNDER FITTING
            Number of training epochs 
            
        Returns:
        --------
        transformers.Trainer
            Trained model
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # Use loss for best model (this uses log loss/cross-entropy internally)
            greater_is_better=False,            # Lower loss is better
            learning_rate=2e-5,                 # Explicitly set learning rate 
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            warmup_steps=500,   # Added warmup steps
            logging_steps=100,  # More frequent logging
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train the model
        print("\nStarting training...")
        trainer.train()
        
        # Save the model
        print("\nSaving model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate
        print("\nEvaluating model...")
        eval_result = trainer.evaluate()
        print(f"Evaluation results: {eval_result}")
        
        return trainer

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Initialize trainer
    trainer = FinancialSentimentTrainer()
    
    # Prepare data
    train_dataset, val_dataset = trainer.prepare_data(df)
    
    # Train model
    trainer.train(train_dataset, val_dataset)