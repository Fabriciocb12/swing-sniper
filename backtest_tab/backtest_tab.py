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

    data['Signal'] = 0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1, 0)
    data['Position'] = data['Signal'].diff()

    # Calculate P&L correctly (make sure it's a single column)
    pct_change = data['Close'].pct_change()  # Percentage change in Close
    data['P&L'] = data['Position'] * pct_change  # Multiply Position by pct_change to calculate P&L

    # Cumulative P&L (Portfolio)
    data['Portfolio'] = (1 + data['P&L']).cumprod()  # Cumulative product of P&L

    return data
