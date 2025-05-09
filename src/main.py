"""
Main module for running the complete trading strategies analysis.
This script orchestrates the data loading, strategy execution, and evaluation.
"""

import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime

# Import project modules
from data_loader import download_data, split_train_test, calculate_returns, plot_prices
from baseline import run_baseline_strategies
from advanced import StateFeatures, run_q_learning
from evaluation import evaluate_strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'trading_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Set up project directories."""
    # Get project root directory
    project_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Define subdirectories
    data_dir = os.path.join(project_dir, 'data')
    results_dir = os.path.join(project_dir, 'results')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return project_dir, data_dir, results_dir

def load_or_download_data(data_dir, force_download=False):
    """
    Load data from cache or download if needed.
    
    Parameters:
        data_dir (str): Directory to store data
        force_download (bool): Whether to force download even if cache exists
        
    Returns:
        tuple: (train_close, train_returns, test_close, test_returns)
    """
    # Define required files
    required_files = ['train_close.pkl', 'test_close.pkl']
    
    # Check if all files exist
    all_files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
    
    if all_files_exist and not force_download:
        logger.info("Loading data from cache...")
        
        # Load data
        train_close = pd.read_pickle(os.path.join(data_dir, 'train_close.pkl'))
        test_close = pd.read_pickle(os.path.join(data_dir, 'test_close.pkl'))
        
        # Calculate returns
        train_returns = calculate_returns(train_close)
        test_returns = calculate_returns(test_close)
    else:
        logger.info("Downloading and processing data...")
        
        # Define assets
        assets = ['SPY', 'TLT', 'VNQ', 'GLD', 'EFA']
        
        # Define date range (10 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now().replace(year=datetime.now().year - 10)).strftime('%Y-%m-%d')
        
        # Download data
        close_prices, _ = download_data(assets, start_date, end_date, data_dir)
        
        # Split data into training (8 years) and testing (2 years)
        train_close, test_close = split_train_test(close_prices, train_years=8)
        
        # Calculate returns
        train_returns = calculate_returns(train_close)
        test_returns = calculate_returns(test_close)
        
        # Save data
        train_close.to_pickle(os.path.join(data_dir, 'train_close.pkl'))
        test_close.to_pickle(os.path.join(data_dir, 'test_close.pkl'))
        
        # Plot prices
        plot_prices(
            close_prices, 
            title='Asset Prices (10 Years)',
            save_path=os.path.join(os.path.dirname(data_dir), 'results', 'asset_prices.png')
        )
    
    return train_close, train_returns, test_close, test_returns

def run_strategies(train_close, train_returns, test_close, test_returns, results_dir, run_baseline=True, run_advanced=True, initial_capital=100000):
    """
    Run trading strategies.
    
    Parameters:
        train_close (DataFrame): Training price data
        train_returns (DataFrame): Training returns data
        test_close (DataFrame): Test price data
        test_returns (DataFrame): Test returns data
        results_dir (str): Directory to save results
        run_baseline (bool): Whether to run baseline strategies
        run_advanced (bool): Whether to run advanced strategies
        initial_capital (float): Initial investment amount
        
    Returns:
        dict: Dictionary with results for each strategy
    """
    results = {}
    
    # Run baseline strategies
    if run_baseline:
        logger.info("Running baseline strategies...")
        baseline_results = run_baseline_strategies(
            train_close, 
            train_returns, 
            test_close, 
            test_returns, 
            results_dir,
            initial_capital
        )
        results.update(baseline_results)
    
    # Run advanced strategies
    if run_advanced:
        logger.info("Running advanced strategies...")
        
        # Extract features
        logger.info("Extracting state features...")
        train_features_extractor = StateFeatures(train_close)
        test_features_extractor = StateFeatures(test_close)
        
        train_features = train_features_extractor.extract_all_features()
        test_features = test_features_extractor.extract_all_features()
        
        # Discretize features
        train_features_discrete = train_features_extractor.discretize_features(train_features)
        test_features_discrete = test_features_extractor.discretize_features(test_features)
        
        # Run Q-learning
        train_portfolio, test_portfolio, policy = run_q_learning(
            train_close, 
            train_features_discrete, 
            test_close, 
            test_features_discrete,
            results_dir,
            initial_capital
        )
        
        # Save results
        train_portfolio.to_pickle(os.path.join(results_dir, 'q_learning_train.pkl'))
        test_portfolio.to_pickle(os.path.join(results_dir, 'q_learning_test.pkl'))
        
        # Add to results
        results['Q-Learning'] = {
            'train': train_portfolio,
            'test': test_portfolio
        }
    
    return results

def main():
    """Main function to run the trading strategies analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trading strategies analysis')
    parser.add_argument('--force-download', action='store_true', help='Force download of data even if cache exists')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline strategies')
    parser.add_argument('--skip-advanced', action='store_true', help='Skip advanced strategies')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    args = parser.parse_args()
    
    # Set up directories
    project_dir, data_dir, results_dir = setup_directories()
    
    # Create log directory
    os.makedirs(os.path.join(results_dir), exist_ok=True)
    
    logger.info("Starting trading strategies analysis...")
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Load or download data
    train_close, train_returns, test_close, test_returns = load_or_download_data(
        data_dir, 
        force_download=args.force_download
    )
    
    # Run strategies
    results = run_strategies(
        train_close, 
        train_returns, 
        test_close, 
        test_returns, 
        results_dir,
        run_baseline=not args.skip_baseline,
        run_advanced=not args.skip_advanced,
        initial_capital=args.capital
    )
    
    # Evaluate strategies
    evaluate_strategies(results, results_dir, initial_capital=args.capital)
    
    logger.info("Trading strategies analysis completed.")

if __name__ == "__main__":
    main()
