import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def fetch_data(ticker):
    # Download data for the ticker for the past 1 year
    data = yf.download(ticker, period="1y", interval="1d")
    return data

def backtest_strategy(data):
    # Example strategy: Moving Average Crossover
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average

    # Generate buy/sell signals
    data['Signal'] = 0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1, 0)
    data['Position'] = data['Signal'].diff()

    # Calculate P&L based on signals
    data['P&L'] = data['Position'] * data['Close'].pct_change()  # Percent change in price
    data['Portfolio'] = (1 + data['P&L']).cumprod()  # Cumulative product of P&L

    return data

def plot_results(data):
    # Plot the portfolio performance (equity curve)
    plt.figure(figsize=(10,6))
    plt.plot(data['Portfolio'], label='Strategy Portfolio', color='blue')
    plt.title('Backtest Results - Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    st.pyplot(plt)

# Streamlit interface
st.title('Swing Sniper GPT - Backtest Your Strategy')

# Sidebar for selecting tickers
tickers = ['AAPL', 'NVDA', 'GOOGL', 'MSFT', 'AMZN']  # Add more tickers as needed
selected_ticker = st.sidebar.selectbox('Select Ticker for Backtest', tickers)

# Add a button to run the backtest
if st.sidebar.button('Run Backtest'):
    st.write(f'Running backtest for {selected_ticker}...')
    data = fetch_data(selected_ticker)  # Get the data for the selected ticker
    backtest_results = backtest_strategy(data)  # Run the backtest
    plot_results(backtest_results)  # Plot the results
