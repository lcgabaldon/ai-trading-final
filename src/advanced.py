"""
Advanced strategies implementation for trading analysis.
This module implements the more sophisticated portfolio allocation strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateFeatures:
    """Class to extract state features from price data."""
    
    def __init__(self, prices):
        """
        Initialize state features.
        
        Parameters:
            prices (DataFrame): DataFrame with price data
        """
        self.prices = prices
        self.assets = prices.columns
        self.scaler = StandardScaler()
    
    def calculate_returns(self, window=1):
        """Calculate returns over a specified window."""
        return self.prices.pct_change(window).fillna(0)
    
    def calculate_volatility(self, window=20):
        """Calculate rolling volatility."""
        returns = self.calculate_returns()
        return returns.rolling(window=window).std().fillna(0)
    
    def calculate_momentum(self, window=20):
        """Calculate momentum (cumulative return over window)."""
        return self.prices.pct_change(window).fillna(0)
    
    def calculate_moving_averages(self, windows=[10, 50, 200]):
        """Calculate moving averages for different windows."""
        ma_features = {}
        for window in windows:
            ma = self.prices.rolling(window=window).mean().fillna(method='bfill')
            # Calculate ratio to current price
            ma_ratio = self.prices / ma - 1
            ma_features[f'ma_{window}'] = ma_ratio
        return ma_features
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        # Fast EMA
        ema_fast = self.prices.ewm(span=fast, adjust=False).mean()
        # Slow EMA
        ema_slow = self.prices.ewm(span=slow, adjust=False).mean()
        # MACD line
        macd_line = ema_fast - ema_slow
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        # MACD histogram
        macd_histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': macd_histogram
        }
    
    def calculate_rsi(self, window=14):
        """Calculate RSI (Relative Strength Index)."""
        # Calculate daily price changes
        delta = self.prices.diff().fillna(0)
        
        # Calculate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean().fillna(0)
        avg_loss = loss.rolling(window=window).mean().fillna(0)
        
        # Calculate relative strength
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(0)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Default to 50 when undefined
        
        return rsi
    
    def extract_all_features(self):
        """Extract all state features."""
        # Basic returns
        returns_1d = self.calculate_returns(window=1)
        returns_5d = self.calculate_returns(window=5)
        returns_20d = self.calculate_returns(window=20)
        
        # Volatility
        volatility = self.calculate_volatility()
        
        # Momentum
        momentum = self.calculate_momentum()
        
        # Moving averages
        ma_features = self.calculate_moving_averages()
        
        # MACD
        macd_features = self.calculate_macd()
        
        # RSI
        rsi = self.calculate_rsi()
        
        # Combine all features
        features = {}
        features['returns_1d'] = returns_1d
        features['returns_5d'] = returns_5d
        features['returns_20d'] = returns_20d
        features['volatility'] = volatility
        features['momentum'] = momentum
        features['rsi'] = rsi
        
        for key, value in ma_features.items():
            features[key] = value
        
        for key, value in macd_features.items():
            features[key] = value
        
        return features
    
    def discretize_features(self, features, num_buckets=3):
        """
        Discretize features into buckets.
        
        Parameters:
            features (dict): Dictionary of feature DataFrames
            num_buckets (int): Number of buckets to use
            
        Returns:
            dict: Dictionary of discretized feature DataFrames
        """
        discretized = {}
        
        for feature_name, feature_df in features.items():
            discretized[feature_name] = pd.DataFrame(index=feature_df.index)
            
            for col in feature_df.columns:
                # Get feature values
                values = feature_df[col].values.reshape(-1, 1)
                
                # Fit scaler on values
                self.scaler.fit(values)
                
                # Transform values
                scaled_values = self.scaler.transform(values).flatten()
                
                # Discretize into buckets
                discretized[feature_name][col] = pd.qcut(
                    scaled_values, 
                    num_buckets, 
                    labels=False, 
                    duplicates='drop'
                ).astype(int)
                
        return discretized

class QLearningStrategy:
    """Class to implement Q-Learning for portfolio allocation."""
    
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize Q-Learning.
        
        Parameters:
            states (list): List of possible states
            actions (list): List of possible actions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table - use a nested dictionary with state hashes as keys
        self.q_table = {}
        for state in states:
            # Convert state tuple to a string representation for hashing
            state_key = self._get_state_key(state)
            self.q_table[state_key] = {}
            for i, action in enumerate(actions):
                # Use action index as key in the inner dictionary
                self.q_table[state_key][i] = 0.0
    
    def _get_state_key(self, state):
        """Convert a state (which could be a tuple containing numpy arrays) to a hashable key."""
        if isinstance(state, tuple):
            # For tuples, recursively convert elements
            return tuple(self._get_state_key(x) for x in state)
        elif isinstance(state, np.ndarray):
            # Convert numpy arrays to tuple for hashability
            return tuple(state.flatten())
        else:
            # Other types should be hashable already
            return state
    
    def _get_action_key(self, action):
        """Convert an action to a hashable key."""
        if isinstance(action, np.ndarray):
            return tuple(action.flatten())
        return action
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Parameters:
            state: Current state
            
        Returns:
            action: Chosen action
        """
        state_key = self._get_state_key(state)
        
        # If state not in Q-table, add it
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}
        
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            action_idx = np.random.choice(len(self.actions))
            return self.actions[action_idx]
        else:
            # Exploit: choose the best action
            action_idx = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return self.actions[action_idx]
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Parameters:
            state: Current state
            action: Chosen action
            reward: Received reward
            next_state: Next state
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Find action index
        action_idx = None
        for i, a in enumerate(self.actions):
            if np.array_equal(a, action):
                action_idx = i
                break
        
        if action_idx is None:
            # Action not found, can't update
            return
        
        # Ensure states exist in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(len(self.actions))}
        
        # Get current Q-value
        current_q = self.q_table[state_key][action_idx]
        
        # Get max Q-value for next state
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Calculate new Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state_key][action_idx] = new_q
    
    def train(self, episodes, env):
        """
        Train the Q-learning agent.
        
        Parameters:
            episodes (int): Number of episodes to train for
            env: Environment to train on
        """
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Choose action
                action = self.choose_action(state)
                
                # Take action
                next_state, reward, done = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                # Update state
                state = next_state
                
                # Update total reward
                total_reward += reward
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{episodes}, Total reward: {total_reward:.4f}")
                
    def get_policy(self):
        """
        Get the learned policy.
        
        Returns:
            dict: Dictionary mapping states to best actions
        """
        policy = {}
        for state_key in self.q_table:
            action_idx = max(self.q_table[state_key], key=self.q_table[state_key].get)
            policy[state_key] = self.actions[action_idx]
        return policy
    
    def save_q_table(self, filename):
        """
        Save Q-table to file.
        
        Parameters:
            filename (str): Filename to save to
        """
        # Convert q_table to serializable format
        serializable_q_table = {}
        for state_key, actions in self.q_table.items():
            serializable_q_table[str(state_key)] = {str(action_idx): value for action_idx, value in actions.items()}
        
        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f)
        
        logger.info(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename):
        """
        Load Q-table from file.
        
        Parameters:
            filename (str): Filename to load from
        """
        with open(filename, 'r') as f:
            serializable_q_table = json.load(f)
        
        # Convert serializable format back to q_table
        self.q_table = {}
        for state_key_str, actions in serializable_q_table.items():
            self.q_table[eval(state_key_str)] = {int(action_idx): value for action_idx, value in actions.items()}
        
        logger.info(f"Q-table loaded from {filename}")

class TradingEnvironment:
    """Class to implement a trading environment for reinforcement learning."""
    
    def __init__(self, prices, features, initial_capital=100000):
        """
        Initialize trading environment.
        
        Parameters:
            prices (DataFrame): DataFrame with price data
            features (dict): Dictionary of state features
            initial_capital (float): Initial investment amount
        """
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            tuple: Initial state
        """
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.done = False
        self.holdings = np.zeros(len(self.prices.columns))
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            tuple: State representation
        """
        # Get current features
        state_features = []
        
        for feature_name, feature_df in self.features.items():
            if self.current_step < len(feature_df):
                state_features.extend(feature_df.iloc[self.current_step].values)
        
        # Convert to tuple for hashability
        return tuple(state_features)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
            action: Action to take (portfolio allocation)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        if self.done:
            return self._get_state(), 0, self.done
        
        # Get current and next prices
        current_prices = self.prices.iloc[self.current_step]
        
        # Apply action (portfolio allocation)
        self.holdings = action
        
        # Move to next step
        self.current_step += 1
        
        # Check if we're done
        if self.current_step >= len(self.prices) - 1:
            self.done = True
        
        # Get next prices
        next_prices = self.prices.iloc[self.current_step]
        
        # Calculate portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + np.sum(self.holdings * (next_prices / current_prices - 1)))
        
        # Calculate reward (daily return)
        reward = (self.portfolio_value / old_portfolio_value) - 1
        
        return self._get_state(), reward, self.done
    
    def render(self):
        """Render the environment."""
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")

def define_actions(num_assets, num_allocations=5):
    """
    Define possible allocation actions.
    
    Parameters:
        num_assets (int): Number of assets
        num_allocations (int): Number of allocation levels
        
    Returns:
        list: List of possible actions (weights arrays)
    """
    # Define allocation levels
    allocation_levels = np.linspace(0, 1, num_allocations)
    
    # Generate all possible allocations
    actions = []
    
    # This is a simplified approach - we're only considering allocations
    # where the weights sum to 1 (fully invested)
    
    # For simplicity, we'll use predefined allocation strategies
    # 1. Equal weight
    equal_weight = np.ones(num_assets) / num_assets
    actions.append(equal_weight)
    
    # 2. 100% in each asset
    for i in range(num_assets):
        weights = np.zeros(num_assets)
        weights[i] = 1.0
        actions.append(weights)
    
    # 3. 60/40 split between each pair of assets
    for i in range(num_assets):
        for j in range(i+1, num_assets):
            weights = np.zeros(num_assets)
            weights[i] = 0.6
            weights[j] = 0.4
            actions.append(weights)
            
            weights = np.zeros(num_assets)
            weights[i] = 0.4
            weights[j] = 0.6
            actions.append(weights)
    
    # 4. Equal weight among 3 assets
    if num_assets >= 3:
        for i in range(num_assets):
            for j in range(i+1, num_assets):
                for k in range(j+1, num_assets):
                    weights = np.zeros(num_assets)
                    weights[i] = 1/3
                    weights[j] = 1/3
                    weights[k] = 1/3
                    actions.append(weights)
    
    return actions

def run_q_learning(train_prices, train_features, test_prices, test_features, results_dir, initial_capital=100000):
    """
    Run Q-learning on training data and evaluate on test data.
    
    Parameters:
        train_prices (DataFrame): Training price data
        train_features (dict): Training state features
        test_prices (DataFrame): Test price data
        test_features (dict): Test state features
        results_dir (str): Directory to save results
        initial_capital (float): Initial investment amount
        
    Returns:
        tuple: (train_portfolio_values, test_portfolio_values, policy)
    """
    logger.info("Running Q-learning strategy...")
    
    # Reduce state complexity for faster learning
    # We'll just use a simplified state representation based on moving averages
    # and discretized into a small number of states
    
    # Define a simplified state representation
    if 'ma_50' in train_features and 'ma_200' in train_features:
        # Use moving average crossover signal as state
        ma_signal_train = (train_features['ma_50'] > 0).astype(int)
        ma_signal_test = (test_features['ma_50'] > 0).astype(int)
        
        # Create simple state tuples (one value per asset)
        simple_train_states = [tuple(row) for _, row in ma_signal_train.iterrows()]
        simple_test_states = [tuple(row) for _, row in ma_signal_test.iterrows()]
    else:
        # If moving averages not available, create random states for demonstration
        np.random.seed(42)
        simple_train_states = [tuple(np.random.randint(0, 2, len(train_prices.columns))) 
                              for _ in range(len(train_prices))]
        simple_test_states = [tuple(np.random.randint(0, 2, len(test_prices.columns))) 
                             for _ in range(len(test_prices))]
    
    # Create training environment with simplified states
    class SimplifiedEnvironment:
        def __init__(self, prices, states, initial_capital):
            self.prices = prices
            self.states = states
            self.initial_capital = initial_capital
            self.reset()
        
        def reset(self):
            self.current_step = 0
            self.portfolio_value = self.initial_capital
            self.done = False
            self.holdings = np.zeros(len(self.prices.columns))
            return self.states[self.current_step]
        
        def step(self, action):
            if self.done:
                return self.states[self.current_step], 0, self.done
            
            # Get current and next prices
            if self.current_step + 1 >= len(self.prices):
                self.done = True
                return self.states[self.current_step], 0, self.done
                
            current_prices = self.prices.iloc[self.current_step]
            
            # Apply action (portfolio allocation)
            self.holdings = action
            
            # Move to next step
            self.current_step += 1
            
            # Get next prices
            next_prices = self.prices.iloc[self.current_step]
            
            # Calculate portfolio value
            old_portfolio_value = self.portfolio_value
            self.portfolio_value = self.portfolio_value * (1 + np.sum(self.holdings * (next_prices / current_prices - 1)))
            
            # Calculate reward (daily return)
            reward = (self.portfolio_value / old_portfolio_value) - 1
            
            return self.states[self.current_step], reward, self.done
    
    # Create environment
    train_env = SimplifiedEnvironment(train_prices, simple_train_states, initial_capital)
    
    # Define actions - simplified for faster learning
    # We'll use 4 basic allocation strategies
    num_assets = len(train_prices.columns)
    actions = [
        np.array([1.0/num_assets] * num_assets),  # Equal weight
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 100% in first asset (typically SPY)
        np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0]),  # 60/40 allocation
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])   # 100% in cash
    ]
    
    # Get unique states
    unique_states = list(set(simple_train_states))
    
    # Initialize Q-learning
    q_learning = QLearningStrategy(unique_states, actions, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # Train Q-learning - use fewer episodes for faster training
    episodes = 10  # Reduced for demonstration
    q_learning.train(episodes, train_env)
    
    # Get policy
    policy = q_learning.get_policy()
    
    # Save Q-table
    q_table_file = os.path.join(results_dir, 'q_table.json')
    q_learning.save_q_table(q_table_file)
    
    # Evaluate policy on training data
    train_portfolio_values = pd.DataFrame(
        index=train_prices.index,
        columns=['Portfolio Value']
    )
    
    # Start with initial capital
    portfolio_value = initial_capital
    train_portfolio_values.iloc[0] = portfolio_value
    
    # Apply policy to each day
    for t in range(1, len(train_prices)):
        state = simple_train_states[t-1]
        state_key = q_learning._get_state_key(state)
        
        # Get action for state or use default
        if state_key in policy:
            action = policy[state_key]
        else:
            action = actions[0]  # Default to equal weight
        
        # Calculate portfolio value
        prev_value = portfolio_value
        returns = train_prices.iloc[t] / train_prices.iloc[t-1] - 1
        portfolio_value = prev_value * (1 + np.sum(action * returns))
        
        # Store portfolio value
        train_portfolio_values.iloc[t] = portfolio_value
    
    # Evaluate policy on test data
    test_portfolio_values = pd.DataFrame(
        index=test_prices.index,
        columns=['Portfolio Value']
    )
    
    # Start with initial capital
    portfolio_value = initial_capital
    test_portfolio_values.iloc[0] = portfolio_value
    
    # Apply policy to each day
    for t in range(1, len(test_prices)):
        if t-1 < len(simple_test_states):
            state = simple_test_states[t-1]
            state_key = q_learning._get_state_key(state)
            
            # Get action for state or use default
            if state_key in policy:
                action = policy[state_key]
            else:
                action = actions[0]  # Default to equal weight
            
            # Calculate portfolio value
            prev_value = portfolio_value
            returns = test_prices.iloc[t] / test_prices.iloc[t-1] - 1
            portfolio_value = prev_value * (1 + np.sum(action * returns))
            
            # Store portfolio value
            test_portfolio_values.iloc[t] = portfolio_value
    
    return train_portfolio_values, test_portfolio_values, policy

def main():
    """Main function to run advanced strategies."""
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if pickle files exist
    required_files = ['train_close.pkl', 'test_close.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        logger.error(f"Missing data files: {missing_files}. Please run data_loader.py first.")
        return
    
    # Load data
    train_close = pd.read_pickle(os.path.join(data_dir, 'train_close.pkl'))
    test_close = pd.read_pickle(os.path.join(data_dir, 'test_close.pkl'))
    
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
        results_dir
    )
    
    # Save results
    train_portfolio.to_pickle(os.path.join(results_dir, 'q_learning_train.pkl'))
    test_portfolio.to_pickle(os.path.join(results_dir, 'q_learning_test.pkl'))
    
    # Print results
    initial_value = test_portfolio.iloc[0, 0]
    final_value = test_portfolio.iloc[-1, 0]
    total_return = (final_value / initial_value - 1) * 100
    annualized_return = ((final_value / initial_value) ** (252 / len(test_portfolio)) - 1) * 100
    
    print("\nQ-Learning Strategy Performance (Test Data):")
    print(f"  Initial Value: ${initial_value:.2f}")
    print(f"  Final Value: ${final_value:.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Annualized Return: {annualized_return:.2f}%")
    
    logger.info("Advanced strategies evaluation completed.")

if __name__ == "__main__":
    main()
