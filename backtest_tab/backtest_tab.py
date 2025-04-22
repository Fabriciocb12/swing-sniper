# backtest_tab.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(ticker):
    # Download data for the ticker for the past 1 year
    data = yf.download(ticker, period="1y", interval="1d")
    return data

def backtest_strategy(data):
    # Example strategy: Moving Average Crossover
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average

    data['Signal'] = 0  # Initialize Signal column to 0
    data.iloc[50:, data.columns.get_loc('Signal')] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1, 0)
    
    data['Position'] = data['Signal'].diff()  # Position change when the signal changes

    # Calculate P&L correctly (ensure it’s a single column)
    pct_change = data['Close'].pct_change()  # Percentage change in Close price
    data['P&L'] = data['Position'] * pct_change  # P&L is the change in price when a position is taken

    # Cumulative P&L (Portfolio performance)
    data['Portfolio'] = (1 + data['P&L']).cumprod()  # Cumulative product of P&L

    return data
