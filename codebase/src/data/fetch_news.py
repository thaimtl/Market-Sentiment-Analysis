import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import traceback
from preprocess import clean_text

def fetch_stock_news_alphavantage(ticker, days=7, api_key='APRTD0XNPCP0J0YC'):
    """
    Fetch news for a given stock ticker using Alpha Vantage API
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., AAPL, MSFT)
    days : int, default=7
        Number of days to look back for news
    api_key : str, default='APRTD0XNPCP0J0YC'
        Alpha Vantage API key (free tier)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing news headlines with dates
    """
    print(f"Fetching news for {ticker} from Alpha Vantage...")
    
    # Get end date as today
    end_date = datetime.now()
    # Get start date as days before today
    start_date = end_date - timedelta(days=days)
    
    try:
        # Create URL
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
        
        # Make request
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            return pd.DataFrame(columns=['Date', 'Headline'])
        
        # Parse JSON
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            print(f"API Error: {data['Error Message']}")
            return pd.DataFrame(columns=['Date', 'Headline'])
        
        # Check for information messages (e.g., API limit reached)
        if 'Information' in data:
            print(f"API Information: {data['Information']}")
            if 'Note' in data:
                print(f"API Note: {data['Note']}")
            return pd.DataFrame(columns=['Date', 'Headline'])
        
        # Extract news items
        news_items = data.get('feed', [])
        if not news_items:
            print(f"No news found for {ticker}")
            return pd.DataFrame(columns=['Date', 'Headline'])
        
        # Process news data
        news_data = []
        
        for item in news_items:
            # Get headline
            headline = item.get('title')
            
            # Get date
            time_published = item.get('time_published')
            if time_published:
                try:
                    # Format: YYYYMMDDTHHMMSS
                    date = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                except ValueError:
                    print(f"Error parsing date: {time_published}")
                    date = datetime.now()
            else:
                date = datetime.now()
            
            # Only include news within the requested date range
            if start_date <= date <= end_date and headline:
                news_data.append({
                    'Date': date,
                    'Headline': headline
                })
        
        # Create dataframe
        news_df = pd.DataFrame(news_data)
        
        # If no news was found in the date range
        if news_df.empty:
            print(f"No news found for {ticker} in the specified date range")
            return news_df
            
        # Clean headlines
        news_df['Cleaned_Headline'] = news_df['Headline'].apply(clean_text)
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
        output_path = os.path.join(output_dir, f"cleaned_data_for_{ticker}__{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv")
        news_df.to_csv(output_path, index=False)
        
        
        print(f"Cleaned data saved to: {output_path}")
        
        print(f"Successfully fetched {len(news_df)} news items for {ticker}")
        
        return news_df
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['Date', 'Headline'])

def fetch_stock_news(ticker, days=7):
    # try Alpha Vantage
    print("trying Alpha Vantage...")
    news_df = fetch_stock_news_alphavantage(ticker, days)
    
    return news_df

if __name__ == "__main__":
    # Test the function
    ticker = "AAPL"
    print(f"Fetching news for {ticker}...")
    news = fetch_stock_news(ticker, days=7)
    
    if not news.empty:
        print(f"Found {len(news)} news items")
        print(news.head())
    else:
        print(f"No news found for {ticker}")