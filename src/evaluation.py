"""
Evaluation module for trading strategies analysis.
This module handles performance evaluation metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_performance_metrics(portfolio_values):
    """
    Calculate performance metrics for a portfolio.
    
    Parameters:
        portfolio_values (DataFrame): DataFrame with portfolio values
        
    Returns:
        dict: Dictionary with performance metrics
    """
    # Extract portfolio values
    values = portfolio_values['Portfolio Value']
    
    # Convert to returns
    returns = values.pct_change().dropna()
    
    # Calculate metrics
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    annualized_return = ((values.iloc[-1] / values.iloc[0]) ** (252 / len(values)) - 1)
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate other metrics
    win_rate = len(returns[returns > 0]) / len(returns)
    loss_rate = len(returns[returns < 0]) / len(returns)
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Loss Rate': loss_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Initial Value': values.iloc[0],
        'Final Value': values.iloc[-1]
    }
    
    return metrics

def calculate_drawdown(portfolio_values):
    """
    Calculate drawdown series for a portfolio.
    
    Parameters:
        portfolio_values (DataFrame): DataFrame with portfolio values
        
    Returns:
        Series: Drawdown series
    """
    # Extract portfolio values
    values = portfolio_values['Portfolio Value']
    
    # Calculate returns
    returns = values.pct_change().fillna(0)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    
    return drawdown

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

def plot_drawdown(portfolios, title='Drawdown Comparison', save_path=None):
    """
    Plot drawdown comparison for different strategies.
    
    Parameters:
        portfolios (dict): Dictionary with strategy names as keys and portfolio values as values
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for strategy, portfolio in portfolios.items():
        drawdown = calculate_drawdown(portfolio)
        plt.plot(drawdown.index, drawdown, label=strategy)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    # Add horizontal lines for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Drawdown comparison plot saved to {save_path}")
    
    plt.close()

def plot_monthly_returns(portfolio_values, strategy_name, save_path=None):
    """
    Plot monthly returns heatmap.
    
    Parameters:
        portfolio_values (DataFrame): DataFrame with portfolio values
        strategy_name (str): Name of the strategy
        save_path (str, optional): Path to save the plot
    """
    # Extract portfolio values
    values = portfolio_values['Portfolio Value']
    
    # Calculate daily returns
    daily_returns = values.pct_change().fillna(0)
    
    # Convert to monthly returns
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table with years as rows and months as columns
    monthly_returns.index = monthly_returns.index.to_period('M')
    monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
    monthly_pivot = monthly_pivot.unstack(level=1)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_pivot, annot=True, fmt=".2%", cmap="RdYlGn", center=0, linewidths=1, cbar=True)
    
    plt.title(f'Monthly Returns Heatmap - {strategy_name}')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Monthly returns heatmap saved to {save_path}")
    
    plt.close()

def create_performance_report(portfolios, test_period=True, save_path=None):
    """
    Create a performance report for multiple strategies.
    
    Parameters:
        portfolios (dict): Dictionary with strategy names as keys and portfolio values as values
        test_period (bool): Whether this is the test period (for title)
        save_path (str, optional): Path to save the report
    """
    # Calculate metrics for each strategy
    metrics = {}
    for strategy, portfolio in portfolios.items():
        metrics[strategy] = calculate_performance_metrics(portfolio)
    
    # Create DataFrames for each metric
    metric_dfs = {}
    for metric in list(list(metrics.values())[0].keys()):
        metric_dfs[metric] = pd.Series({strategy: metrics[strategy][metric] for strategy in metrics.keys()})
    
    # Combine key metrics into a single DataFrame
    key_metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Maximum Drawdown']
    performance_df = pd.DataFrame({metric: metric_dfs[metric] for metric in key_metrics})
    
    # Format the DataFrame
    performance_df['Total Return'] = performance_df['Total Return'].map('{:.2%}'.format)
    performance_df['Annualized Return'] = performance_df['Annualized Return'].map('{:.2%}'.format)
    performance_df['Volatility'] = performance_df['Volatility'].map('{:.2%}'.format)
    performance_df['Sharpe Ratio'] = performance_df['Sharpe Ratio'].map('{:.2f}'.format)
    performance_df['Maximum Drawdown'] = performance_df['Maximum Drawdown'].map('{:.2%}'.format)
    
    # Print the report
    period = "Test Period" if test_period else "Training Period"
    print(f"\nPerformance Report ({period}):")
    print(performance_df.to_string())
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"Performance Report ({period})\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(performance_df.to_string())
        logger.info(f"Performance report saved to {save_path}")
    
    return performance_df

def evaluate_strategies(results, results_dir, initial_capital=100000):
    """
    Evaluate strategies and generate reports and visualizations.
    
    Parameters:
        results (dict): Dictionary with results for each strategy
        results_dir (str): Directory to save results
        initial_capital (float): Initial investment amount
        
    Returns:
        tuple: (train_report, test_report)
    """
    logger.info("Evaluating strategies...")
    
    # Create dictionaries for portfolio values
    train_portfolios = {strategy: results[strategy]['train'] for strategy in results.keys()}
    test_portfolios = {strategy: results[strategy]['test'] for strategy in results.keys()}
    
    # Create performance reports
    train_report = create_performance_report(
        train_portfolios, 
        test_period=False,
        save_path=os.path.join(results_dir, 'train_performance_report.txt')
    )
    
    test_report = create_performance_report(
        test_portfolios, 
        test_period=True,
        save_path=os.path.join(results_dir, 'test_performance_report.txt')
    )
    
    # Plot portfolio comparisons
    plot_portfolio_comparison(
        train_portfolios, 
        title='Strategy Comparison (Training Data)',
        save_path=os.path.join(results_dir, 'portfolio_comparison_train.png')
    )
    
    plot_portfolio_comparison(
        test_portfolios, 
        title='Strategy Comparison (Test Data)',
        save_path=os.path.join(results_dir, 'portfolio_comparison_test.png')
    )
    
    # Plot drawdowns
    plot_drawdown(
        train_portfolios, 
        title='Drawdown Comparison (Training Data)',
        save_path=os.path.join(results_dir, 'drawdown_comparison_train.png')
    )
    
    plot_drawdown(
        test_portfolios, 
        title='Drawdown Comparison (Test Data)',
        save_path=os.path.join(results_dir, 'drawdown_comparison_test.png')
    )
    
    # Helper function to sanitize filenames
    def sanitize_filename(filename):
        # Replace invalid characters with underscores
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
    
    # Plot monthly returns for each strategy
    for strategy in results.keys():
        sanitized_strategy_name = sanitize_filename(strategy)
        filename = f'monthly_returns_{sanitized_strategy_name.lower()}.png'
        plot_monthly_returns(
            test_portfolios[strategy], 
            strategy,
            save_path=os.path.join(results_dir, filename)
        )
    
    return train_report, test_report

def main():
    """Main function to evaluate strategies."""
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if pickle files exist for strategies
    baseline_strategies = ['Random Allocation', '60/40 Allocation', 'Max Sharpe', 'Moving Average']
    advanced_strategies = ['Q-Learning']
    
    # Load baseline strategies
    baseline_results = {}
    for strategy in baseline_strategies:
        strategy_key = strategy.replace(' ', '_').lower()
        train_file = os.path.join(results_dir, f'{strategy_key}_train.pkl')
        test_file = os.path.join(results_dir, f'{strategy_key}_test.pkl')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            baseline_results[strategy] = {
                'train': pd.read_pickle(train_file),
                'test': pd.read_pickle(test_file)
            }
        else:
            logger.warning(f"Missing result files for {strategy}. Run baseline.py first.")
    
    # Load advanced strategies
    advanced_results = {}
    for strategy in advanced_strategies:
        strategy_key = strategy.replace(' ', '_').lower()
        train_file = os.path.join(results_dir, f'{strategy_key}_train.pkl')
        test_file = os.path.join(results_dir, f'{strategy_key}_test.pkl')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            advanced_results[strategy] = {
                'train': pd.read_pickle(train_file),
                'test': pd.read_pickle(test_file)
            }
        else:
            logger.warning(f"Missing result files for {strategy}. Run advanced.py first.")
    
    # Combine results
    all_results = {**baseline_results, **advanced_results}
    
    # If no results found, exit
    if not all_results:
        logger.error("No strategy results found. Run baseline.py and advanced.py first.")
        return
    
    # Evaluate strategies
    train_report, test_report = evaluate_strategies(all_results, results_dir)
    
    logger.info("Strategy evaluation completed.")

if __name__ == "__main__":
    main()
