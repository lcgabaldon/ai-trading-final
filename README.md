# Trading Analysis and Strategies

This project implements and evaluates various portfolio allocation strategies using historical data from different asset classes.

## Project Overview

The project analyzes the performance of different trading and portfolio allocation strategies using 10 years of historical data from 5 broad asset class indexes. The data is split into 8 years for training and 2 years for evaluation.

### Asset Classes

The project uses the following asset classes:
1. Equities: SPY (S&P 500 ETF)
2. Bonds: TLT (Long-term Treasury ETF)
3. Real Estate: VNQ (Vanguard Real Estate ETF)
4. Commodities: GLD (Gold ETF)
5. International: EFA (MSCI EAFE ETF)
6. Cash (zero-return option)

### Implemented Strategies

#### Baseline Strategies
1. Uniform Random Allocation: Random allocation across the 5 investable assets
2. 60/40 Allocation: Standard allocation with 60% to equities and 40% to bonds
3. Sharpe Ratio Optimization: Portfolio that maximizes the Sharpe ratio on training data
4. Moving Average Strategy: Trading strategy based on 50 and 200-day moving averages

#### Advanced Strategies
- Reinforcement Learning (Q-learning) based strategies
- Additional state features and optimization techniques

## Project Structure

```
trading-strategies-analysis/
├── data/                    # For storing downloaded asset data
├── notebooks/               # For analysis and strategy development
├── src/                     # Source code for strategy implementations
│   ├── __init__.py
│   ├── data_loader.py       # Functions to download and process data
│   ├── baseline.py          # Baseline strategy implementations
│   ├── advanced.py          # Advanced strategy implementations
│   └── evaluation.py        # Performance evaluation functions
├── results/                 # Store performance results and charts
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Getting Started

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the data collection script: `python src/data_loader.py`
4. Execute strategy analysis: `python src/main.py`
5. View results in the `results` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
