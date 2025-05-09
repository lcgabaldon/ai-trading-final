"""
Data loading module for trading strategies analysis.
This module handles downloading and processing financial data.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the assets
ASSETS = {
    'SPY': 'Equities',       # S&P 500 ETF
    'TLT': 'Bonds',          # Long-term Treasury ETF
    'VNQ': 'Real Estate',    # Vanguard Real Estate ETF
    'GLD': 'Commodities',    # Gold ETF
    'EFA': 'International'   # MSCI EAFE ETF
}

# Add cash (represented as zero return)
RISK_FREE_RATE = 0.02  # Assumed annual risk-free rate (2%)

def generate_sample_data(tickers, start_date, end_date):
    """
    Generate sample price data when API fails.
    
    Parameters:
        tickers (list): List of ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        tuple: DataFrames for closing prices and opening prices
    """
    logger.info(f"Generating sample data for {tickers} from {start_date} to {end_date}")
    
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Create DataFrames
    closing_prices = pd.DataFrame(index=date_range)
    opening_prices = pd.DataFrame(index=date_range)
    
    # Asset parameters (mean, volatility, correlation with SPY)
    params = {
        'SPY': {'mean': 0.0003, 'vol': 0.01, 'corr': 1.0},
        'TLT': {'mean': 0.0001, 'vol': 0.007, 'corr': -0.3},
        'VNQ': {'mean': 0.0002, 'vol': 0.012, 'corr': 0.7},
        'GLD': {'mean': 0.0001, 'vol': 0.009, 'corr': -0.1},
        'EFA': {'mean': 0.00025, 'vol': 0.011, 'corr': 0.8}
    }
    
    # Generate SPY first (as the market factor)
    np.random.seed(42)  # For reproducibility
    spy_returns = np.random.normal(params['SPY']['mean'], params['SPY']['vol'], len(date_range))
    
    # Start with price 100
    spy_prices = 100 * (1 + spy_returns).cumprod()
    closing_prices['SPY'] = spy_prices
    
    # Generate other assets with correlation to SPY
    for ticker in tickers:
        if ticker != 'SPY' and ticker != 'CASH':
            # Generate correlated returns
            uncorr_returns = np.random.normal(0, 1, len(date_range))
            corr_returns = params[ticker]['corr'] * spy_returns + np.sqrt(1 - params[ticker]['corr']**2) * uncorr_returns
            
            # Scale to desired mean and volatility
            asset_returns = corr_returns * params[ticker]['vol'] / np.std(corr_returns) + params[ticker]['mean']
            
            # Convert to prices
            asset_prices = 100 * (1 + asset_returns).cumprod()
            closing_prices[ticker] = asset_prices
    
    # Generate opening prices (slight variation from previous day's close)
    for ticker in tickers:
        if ticker != 'CASH':
            prev_close = closing_prices[ticker].shift(1).fillna(100)
            random_offset = np.random.normal(0, 0.003, len(date_range))
            opening_prices[ticker] = prev_close * (1 + random_offset)
    
    # Add cash
    closing_prices['CASH'] = 1.0
    opening_prices['CASH'] = 1.0
    
    # Calculate daily returns based on cash rate
    daily_rate = (1 + RISK_FREE_RATE) ** (1/252) - 1
    days = np.arange(len(date_range))
    closing_prices['CASH'] = (1 + daily_rate) ** days
    opening_prices['CASH'] = (1 + daily_rate) ** days
    
    return closing_prices, opening_prices

def download_data(tickers, start_date, end_date, data_dir):
    """
    Download historical price data for a list of tickers.
    If download fails, generate sample data.
    
    Parameters:
        tickers (list): List of ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        data_dir (str): Directory to save data
        
    Returns:
        tuple: DataFrames for closing prices and opening prices
    """
    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}")
    
    try:
        # Create cache file path
        os.makedirs(data_dir, exist_ok=True)
        cache_file = os.path.join(data_dir, f"prices_{start_date}_to_{end_date}.pkl")
        
        # Check if data is already cached
        if os.path.exists(cache_file):
            logger.info(f"Loading data from cache: {cache_file}")
            data = pd.read_pickle(cache_file)
            closing_prices = data['Close']
            opening_prices = data['Open']
        else:
            try:
                # Try to download data
                data = yf.download(tickers, start=start_date, end=end_date)
                
                # If data is empty, raise exception
                if len(data) == 0 or data.empty:
                    raise Exception("No data downloaded")
                
                # Extract close and open prices
                closing_prices = data['Adj Close']
                opening_prices = data['Open']
                
                # Add cash as a column with constant value 1.0
                closing_prices['CASH'] = 1.0
                opening_prices['CASH'] = 1.0
                
                # Calculate daily returns based on cash rate
                daily_rate = (1 + RISK_FREE_RATE) ** (1/252) - 1
                days = (closing_prices.index - closing_prices.index[0]).days
                closing_prices['CASH'] = (1 + daily_rate) ** days
                opening_prices['CASH'] = (1 + daily_rate) ** days
            except Exception as e:
                logger.warning(f"Failed to download data: {e}. Generating sample data instead.")
                closing_prices, opening_prices = generate_sample_data(tickers, start_date, end_date)
            
            # Save to cache
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pd.to_pickle({'Close': closing_prices, 'Open': opening_prices}, cache_file)
            logger.info(f"Data saved to cache: {cache_file}")
        
        return closing_prices, opening_prices
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        logger.info("Falling back to sample data generation")
        
        closing_prices, opening_prices = generate_sample_data(tickers, start_date, end_date)
        
        # Save to cache
        cache_file = os.path.join(data_dir, f"sample_prices_{start_date}_to_{end_date}.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        pd.to_pickle({'Close': closing_prices, 'Open': opening_prices}, cache_file)
        logger.info(f"Sample data saved to cache: {cache_file}")
        
        return closing_prices, opening_prices

def split_train_test(data, train_years=8):
    """
    Split data into training and testing sets.
    
    Parameters:
        data (DataFrame): DataFrame with price data
        train_years (int): Number of years for training
        
    Returns:
        tuple: Training and testing DataFrames
    """
    # Calculate the split point based on date
    total_days = len(data)
    train_days = int(total_days * (train_years / 10))
    
    train_data = data.iloc[:train_days]
    test_data = data.iloc[train_days:]
    
    logger.info(f"Data split: Training from {train_data.index[0]} to {train_data.index[-1]}")
    logger.info(f"Data split: Testing from {test_data.index[0]} to {test_data.index[-1]}")
    
    return train_data, test_data

def calculate_returns(prices):
    """
    Calculate daily returns from prices.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        
    Returns:
        DataFrame: Daily returns
    """
    returns = prices.pct_change().dropna()
    return returns

def calculate_stats(returns):
    """
    Calculate key statistics from returns.
    
    Parameters:
        returns (DataFrame): DataFrame with returns data
        
    Returns:
        dict: Dictionary with statistics
    """
    # Annualized return and volatility
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol
    
    # Create a stats dictionary
    stats = {
        'Annual Return': ann_return,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe
    }
    
    return stats

def plot_prices(prices, title='Asset Prices', save_path=None):
    """
    Plot the prices of assets.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    # Normalize prices to start at 100
    normalized = prices.div(prices.iloc[0]) * 100
    
    plt.figure(figsize=(12, 8))
    
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (normalized to 100)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def main():
    """Main function to download and process data."""
    # Define parameters
    tickers = list(ASSETS.keys())
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Download data
    closing_prices, opening_prices = download_data(tickers, start_date, end_date, data_dir)
    
    # Split data
    train_close, test_close = split_train_test(closing_prices)
    train_open, test_open = split_train_test(opening_prices)
    
    # Calculate returns
    train_returns = calculate_returns(train_close)
    test_returns = calculate_returns(test_close)
    
    # Calculate statistics
    train_stats = calculate_stats(train_returns)
    test_stats = calculate_stats(test_returns)
    
    # Print statistics
    print("\nTraining Data Statistics:")
    for stat, values in train_stats.items():
        print(f"\n{stat}:")
        for asset, value in values.items():
            print(f"  {asset}: {value:.4f}")
    
    print("\nTest Data Statistics:")
    for stat, values in test_stats.items():
        print(f"\n{stat}:")
        for asset, value in values.items():
            print(f"  {asset}: {value:.4f}")
    
    # Plot prices
    plot_prices(closing_prices, title='Asset Prices (10 Years)', 
               save_path=os.path.join(results_dir, 'asset_prices.png'))
    
    # Save processed data
    train_close.to_pickle(os.path.join(data_dir, 'train_close.pkl'))
    test_close.to_pickle(os.path.join(data_dir, 'test_close.pkl'))
    train_open.to_pickle(os.path.join(data_dir, 'train_open.pkl'))
    test_open.to_pickle(os.path.join(data_dir, 'test_open.pkl'))
    
    logger.info("Data processing completed and saved.")

if __name__ == "__main__":
    main()
