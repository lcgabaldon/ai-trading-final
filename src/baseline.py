"""
Baseline strategies implementation for trading analysis.
This module implements the required baseline portfolio allocation strategies.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_portfolio_value(prices, weights, initial_capital=100000):
    """
    Calculate portfolio value over time given prices and weights.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        weights (array): Array of portfolio weights
        initial_capital (float): Initial investment amount
        
    Returns:
        DataFrame: Portfolio value over time
    """
    # Normalize prices to get returns
    normalized_prices = prices.div(prices.iloc[0])
    
    # Calculate portfolio value over time (buy and hold)
    weighted_returns = normalized_prices.mul(weights, axis=1)
    portfolio_value = weighted_returns.sum(axis=1) * initial_capital
    
    return pd.DataFrame(portfolio_value, index=prices.index, columns=['Portfolio Value'])

def random_allocation(prices, initial_capital=100000, seed=42):
    """
    Implement random allocation strategy.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        initial_capital (float): Initial investment amount
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: Portfolio values DataFrame and weights array
    """
    np.random.seed(seed)
    assets = prices.columns
    
    # Generate random weights
    weights = np.random.random(len(assets))
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    logger.info(f"Random allocation weights: {dict(zip(assets, weights))}")
    
    # Calculate portfolio value over time
    portfolio_values = calculate_portfolio_value(prices, weights, initial_capital)
    
    return portfolio_values, weights

def sixty_forty_allocation(prices, initial_capital=100000):
    """
    Implement 60/40 equity/bond allocation.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        initial_capital (float): Initial investment amount
        
    Returns:
        tuple: Portfolio values DataFrame and weights array
    """
    assets = prices.columns
    weights = np.zeros(len(assets))
    
    # Find indices for SPY (equities) and TLT (bonds)
    spy_idx = list(prices.columns).index('SPY')
    tlt_idx = list(prices.columns).index('TLT')
    
    # Set weights
    weights[spy_idx] = 0.6  # SPY (equities)
    weights[tlt_idx] = 0.4  # TLT (bonds)
    
    logger.info(f"60/40 allocation weights: {dict(zip(assets, weights))}")
    
    # Calculate portfolio value over time
    portfolio_values = calculate_portfolio_value(prices, weights, initial_capital)
    
    return portfolio_values, weights

def negative_sharpe(weights, returns, risk_free_rate=0.02/252):
    """
    Calculate negative Sharpe ratio for optimization.
    
    Parameters:
        weights (array): Array of portfolio weights
        returns (DataFrame): DataFrame with returns data
        risk_free_rate (float): Daily risk-free rate
        
    Returns:
        float: Negative Sharpe ratio
    """
    # Convert returns to numpy array
    returns_array = returns.values
    
    # Calculate portfolio return and volatility
    port_return = np.sum(returns_array.mean(axis=0) * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns_array, rowvar=False) * 252, weights)))
    
    # Calculate Sharpe ratio
    sharpe = (port_return - risk_free_rate * 252) / port_vol
    
    # Return negative Sharpe for minimization
    return -sharpe

def optimize_sharpe(returns, constraint_set=(0, 1), risk_free_rate=0.02/252):
    """
    Optimize portfolio weights for maximum Sharpe ratio.
    
    Parameters:
        returns (DataFrame): DataFrame with returns data
        constraint_set (tuple): Bounds for weights
        risk_free_rate (float): Daily risk-free rate
        
    Returns:
        array: Optimized weights
    """
    num_assets = len(returns.columns)
    assets = returns.columns
    
    # Initial guess (equal weights)
    initial_guess = np.array([1.0/num_assets] * num_assets)
    
    # Constraints (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds (each weight between constraint_set[0] and constraint_set[1])
    bounds = tuple(constraint_set for _ in range(num_assets))
    
    # Optimize
    args = (returns, risk_free_rate)
    result = minimize(negative_sharpe, initial_guess, args=args,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Check if optimization was successful
    if result['success']:
        optimized_weights = result['x']
        logger.info(f"Sharpe optimization successful. Sharpe: {-result['fun']:.4f}")
        logger.info(f"Optimized weights: {dict(zip(assets, optimized_weights))}")
        return optimized_weights
    else:
        logger.warning(f"Sharpe optimization failed: {result['message']}")
        return initial_guess

def max_sharpe_allocation(prices, returns, initial_capital=100000):
    """
    Implement maximum Sharpe ratio allocation.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        returns (DataFrame): DataFrame with returns data
        initial_capital (float): Initial investment amount
        
    Returns:
        tuple: Portfolio values DataFrame and weights array
    """
    # Optimize weights for maximum Sharpe ratio
    weights = optimize_sharpe(returns)
    
    # Calculate portfolio value over time
    portfolio_values = calculate_portfolio_value(prices, weights, initial_capital)
    
    return portfolio_values, weights

def calculate_moving_averages(prices, short_window=50, long_window=200):
    """
    Calculate moving averages for each asset.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        short_window (int): Short-term moving average window
        long_window (int): Long-term moving average window
        
    Returns:
        tuple: DataFrames for short and long moving averages
    """
    # Calculate moving averages
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    return short_ma, long_ma

def moving_average_strategy(prices, returns, short_window=50, long_window=200, initial_capital=100000):
    """
    Implement moving average crossover strategy.
    
    Parameters:
        prices (DataFrame): DataFrame with price data
        returns (DataFrame): DataFrame with returns data
        short_window (int): Short-term moving average window
        long_window (int): Long-term moving average window
        initial_capital (float): Initial investment amount
        
    Returns:
        tuple: Portfolio values DataFrame and final weights array
    """
    # Calculate moving averages
    short_ma, long_ma = calculate_moving_averages(prices, short_window, long_window)
    
    # Create signals: 1 for long (short MA > long MA), 0 for cash (short MA < long MA)
    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for col in prices.columns:
        signals[col] = 0.0
        # Create signals based on moving average crossover
        signals.loc[short_ma[col] > long_ma[col], col] = 1.0
    
    # Fill NaN values with 0 (no position until both MAs are available)
    signals = signals.fillna(0.0)
    
    # Calculate daily positions (portfolio value for each asset)
    # For each day, allocate capital equally among assets with positive signals
    portfolio_value = pd.Series(initial_capital, index=prices.index)
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Initialize first day positions
    first_valid_idx = max(long_window, 0)
    if first_valid_idx >= len(prices):
        first_valid_idx = len(prices) - 1
        
    for col in prices.columns:
        positions.iloc[first_valid_idx, prices.columns.get_loc(col)] = signals.iloc[first_valid_idx, signals.columns.get_loc(col)] * portfolio_value.iloc[first_valid_idx] * (1.0 / len(prices.columns))
    
    # Calculate daily position values
    for t in range(first_valid_idx + 1, len(prices)):
        # Make sure the indices align between positions and returns
        if t-1 < len(positions) and t < len(returns):
            # Update portfolio value based on yesterday's positions
            prev_positions = positions.iloc[t-1]
            current_returns = returns.iloc[t-1]  # Use t-1 for returns to align with positions
            portfolio_value.iloc[t] = np.sum(prev_positions * (1 + current_returns))
            
            # Count assets with buy signal today
            active_assets = signals.iloc[t].sum()
            
            if active_assets > 0:
                # Allocate today's portfolio equally among assets with buy signal
                for col in prices.columns:
                    # If buy signal, allocate portion of portfolio
                    if signals.iloc[t, signals.columns.get_loc(col)] == 1.0:
                        positions.iloc[t, positions.columns.get_loc(col)] = portfolio_value.iloc[t] / active_assets
                    else:
                        positions.iloc[t, positions.columns.get_loc(col)] = 0.0
            else:
                # If no buy signals, allocate to cash (not included in positions)
                positions.iloc[t] = 0.0
    
    # Calculate final weights based on last day
    final_weights = positions.iloc[-1] / portfolio_value.iloc[-1] if portfolio_value.iloc[-1] > 0 else np.zeros(len(prices.columns))
    
    # Convert portfolio value to DataFrame
    portfolio_df = pd.DataFrame(portfolio_value, columns=['Portfolio Value'])
    
    logger.info(f"Moving average strategy completed. Final portfolio value: {portfolio_df.iloc[-1, 0]:.2f}")
    
    return portfolio_df, final_weights

def plot_portfolio_comparison(portfolios, title='Portfolio Comparison', save_path=None):
    """
    Plot comparison of different portfolio strategies.
    
    Parameters:
        portfolios (dict): Dictionary with strategy names as keys and portfolio values as values
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for strategy, portfolio in portfolios.items():
        # Normalize to initial value
        normalized = portfolio / portfolio.iloc[0] * 100
        plt.plot(normalized.index, normalized['Portfolio Value'], label=strategy)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (normalized to 100)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Portfolio comparison plot saved to {save_path}")
    
    plt.close()

def run_baseline_strategies(train_prices, train_returns, test_prices, test_returns, results_dir, initial_capital=100000):
    """
    Run all baseline strategies on training and test data.
    
    Parameters:
        train_prices (DataFrame): Training price data
        train_returns (DataFrame): Training returns data
        test_prices (DataFrame): Test price data
        test_returns (DataFrame): Test returns data
        results_dir (str): Directory to save results
        initial_capital (float): Initial investment amount
        
    Returns:
        dict: Dictionary with results for each strategy
    """
    logger.info("Running baseline strategies...")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # 1. Random Allocation
    logger.info("Running Random Allocation strategy...")
    random_train, random_weights = random_allocation(train_prices, initial_capital)
    
    # Apply same weights to test data
    random_test = calculate_portfolio_value(test_prices, random_weights, initial_capital)
    
    results['Random Allocation'] = {
        'train': random_train,
        'test': random_test,
        'weights': random_weights
    }
    
    # 2. 60/40 Allocation
    logger.info("Running 60/40 Allocation strategy...")
    sixty_forty_train, sixty_forty_weights = sixty_forty_allocation(train_prices, initial_capital)
    
    # Apply same weights to test data
    sixty_forty_test = calculate_portfolio_value(test_prices, sixty_forty_weights, initial_capital)
    
    results['60/40 Allocation'] = {
        'train': sixty_forty_train,
        'test': sixty_forty_test,
        'weights': sixty_forty_weights
    }
    
    # 3. Maximum Sharpe Ratio
    logger.info("Running Maximum Sharpe Ratio strategy...")
    max_sharpe_train, max_sharpe_weights = max_sharpe_allocation(train_prices, train_returns, initial_capital)
    
    # Apply same weights to test data
    max_sharpe_test = calculate_portfolio_value(test_prices, max_sharpe_weights, initial_capital)
    
    results['Max Sharpe'] = {
        'train': max_sharpe_train,
        'test': max_sharpe_test,
        'weights': max_sharpe_weights
    }
    
    # 4. Moving Average Strategy
    logger.info("Running Moving Average strategy...")
    ma_train, ma_weights = moving_average_strategy(train_prices, train_returns, initial_capital=initial_capital)
    
    # Apply strategy to test data
    ma_test, _ = moving_average_strategy(test_prices, test_returns, initial_capital=initial_capital)
    
    results['Moving Average'] = {
        'train': ma_train,
        'test': ma_test,
        'weights': ma_weights
    }
    
    # Plot comparison of strategies (training data)
    train_portfolios = {
        strategy: results[strategy]['train'] for strategy in results.keys()
    }
    plot_portfolio_comparison(
        train_portfolios, 
        title='Baseline Strategies Comparison (Training Data)',
        save_path=os.path.join(results_dir, 'baseline_comparison_train.png')
    )
    
    # Plot comparison of strategies (test data)
    test_portfolios = {
        strategy: results[strategy]['test'] for strategy in results.keys()
    }
    plot_portfolio_comparison(
        test_portfolios, 
        title='Baseline Strategies Comparison (Test Data)',
        save_path=os.path.join(results_dir, 'baseline_comparison_test.png')
    )
    
    return results

def main():
    """Main function to run baseline strategies."""
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if pickle files exist
    required_files = ['train_close.pkl', 'test_close.pkl', 'train_open.pkl', 'test_open.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        logger.error(f"Missing data files: {missing_files}. Please run data_loader.py first.")
        return
    
    # Load data
    train_close = pd.read_pickle(os.path.join(data_dir, 'train_close.pkl'))
    test_close = pd.read_pickle(os.path.join(data_dir, 'test_close.pkl'))
    
    # Calculate returns
    train_returns = train_close.pct_change().dropna()
    test_returns = test_close.pct_change().dropna()
    
    # Run baseline strategies
    results = run_baseline_strategies(
        train_close[1:],  # Skip first row to align with returns
        train_returns, 
        test_close[1:],   # Skip first row to align with returns
        test_returns,
        results_dir
    )
    
    # Print results summary
    print("\nBaseline Strategies Performance Summary (Test Data):")
    for strategy, data in results.items():
        test_portfolio = data['test']
        initial_value = test_portfolio.iloc[0, 0]
        final_value = test_portfolio.iloc[-1, 0]
        total_return = (final_value / initial_value - 1) * 100
        annualized_return = ((final_value / initial_value) ** (252 / len(test_portfolio)) - 1) * 100
        
        print(f"\n{strategy}:")
        print(f"  Initial Value: ${initial_value:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Annualized Return: {annualized_return:.2f}%")
        
        # Print weights
        print("  Asset Allocation:")
        weights = data['weights']
        assets = train_close.columns
        for asset, weight in zip(assets, weights):
            print(f"    {asset}: {weight:.2%}")
    
    logger.info("Baseline strategies evaluation completed.")

if __name__ == "__main__":
    main()
