import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import os
warnings.filterwarnings('ignore')
# Ensure NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def clean_text(text):
    """Clean financial text while preserving important symbols"""
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def load_and_preprocess_data():
    """Load and preprocess financial sentiment dataset"""
    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the data file
    filepath = os.path.join(current_dir, "..", "..", "data", "raw", "data.csv")
    df = pd.read_csv(filepath)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Basic cleaning
    df['processed_text'] = df['Sentence'].apply(clean_text)
    
    # Check for duplicates before removing them
    duplicates = df.duplicated(subset=['processed_text']).sum()
    print(f"Found {duplicates} duplicates in the dataset")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['processed_text']).reset_index(drop=True)
    
    # Calculate and display the maximum character length in the processed_text column
    max_char_length = df['processed_text'].str.len().max()
    print(f"Maximum character length in processed_text column: {max_char_length}")

    # Map numerical labels to text
    sentiment_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    df['sentiment_text'] = df['Sentiment'].map(sentiment_map)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
    output_path = os.path.join(output_dir, "processed_kagggle_dataset_for_training.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned and processed data saved to: {output_path}")
    return df

def compare_raw_and_cleaned(df, num_samples=5):
    """Compare raw and cleaned text for a sample of rows"""
    print("\n--- Comparing Raw vs Cleaned Text ---")
    
    # Count rows where text was actually modified
    modified_count = (df['Sentence'] != df['processed_text']).sum()
    percent_modified = (modified_count / len(df)) * 100
    
    print(f"Text cleaning modified {modified_count} out of {len(df)} rows ({percent_modified:.2f}%)")
    
    # Show examples of changes
    changed_df = df[df['Sentence'] != df['processed_text']].sample(min(num_samples, modified_count))
    
    if not changed_df.empty:
        for i, row in changed_df.iterrows():
            print("\nExample of cleaning:")
            print(f"Original: \"{row['Sentence']}\"")
            print(f"Cleaned:  \"{row['processed_text']}\"")
            
            # Highlight what was removed
            original_len = len(row['Sentence'])
            cleaned_len = len(row['processed_text'])
            chars_removed = original_len - cleaned_len
            print(f"Characters removed: {chars_removed}")
    else:
        print("No examples of modified text found.")







if __name__ == "__main__":
    download_nltk_resources()
    df = load_and_preprocess_data()
    print(f"Loaded and preprocessed {len(df)} samples.")
    print(f"Sentiment distribution: {df['Sentiment'].value_counts().to_dict()}")
    compare_raw_and_cleaned(df)