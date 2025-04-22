import yfinance as yf
import pandas as pd
import ta
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from openai import OpenAI

# Load environment variables (API keys, email credentials)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

client = OpenAI(api_key=OPENAI_API_KEY)

# Function to send email notifications
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        st.success("📬 Email alert sent!")
    except Exception as e:
        st.error(f"📪 Email failed: {e}")

# Function to fetch data for the selected ticker
def fetch_data(ticker, period="3mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for {ticker}: {e}")
        return None

# Backtest Strategy Function
def backtest_strategy(data):
    # Simple Moving Average Strategy (50-day and 200-day crossover)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Signal'] = 0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1, 0)
    data['Position'] = data['Signal'].diff()
    pct_change = data['Close'].pct_change()
    data['P&L'] = data['Position'] * pct_change
    data['Portfolio'] = (1 + data['P&L']).cumprod()
    return data

# Backtest Results Plot
def plot_results(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Portfolio'], label='Portfolio Value')
    plt.title('Backtest Results: Portfolio Growth')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    st.pyplot()

# Function to scan for high-confidence trades
def scan_for_trades(tickers, confidence_threshold=50):
    triggered_trades = []
    for ticker in tickers:
        data = fetch_data(ticker)
        if data is not None:
            backtest_results = backtest_strategy(data)
            # Add confidence logic here
            trade_confidence = calculate_trade_confidence(backtest_results)
            if trade_confidence >= confidence_threshold:
                triggered_trades.append({
                    "ticker": ticker,
                    "confidence": trade_confidence,
                    "data": backtest_results
                })
    return triggered_trades

# Function to calculate trade confidence (example)
def calculate_trade_confidence(data):
    # This is a simple placeholder for confidence calculation, can be more complex
    if data['P&L'].iloc[-1] > 0:
        return 75  # Example confidence level (you can base it on more factors)
    return 50  # Default low confidence

# Function to run backtest on selected ticker from the sidebar
def run_backtest():
    st.sidebar.title("Backtest Strategy")
    selected_ticker_backtest = st.sidebar.selectbox("Select Ticker for Backtest", final_tickers)

    if st.sidebar.button("Run Backtest"):
        st.write(f"Running backtest for {selected_ticker_backtest}...")

        data = fetch_data(selected_ticker_backtest)
        if data is not None:
            # Run backtest strategy
            backtest_results = backtest_strategy(data)
            plot_results(backtest_results)

# Function to handle trade logging for future training
def log_trade(trade_data):
    log_file = "trade_logs.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            trade_logs = json.load(f)
    else:
        trade_logs = []

    trade_logs.append(trade_data)

    with open(log_file, "w") as f:
        json.dump(trade_logs, f, indent=4)

# Main Logic
def main():
    # Define tickers to scan
    final_tickers = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP"]
    
    # User-defined confidence threshold slider
    confidence_threshold = st.slider("Set Trade Confidence Threshold %", 1, 100, 50, step=1)
    trade_button = st.button("🔍 Scan for Trades")
    status_placeholder = st.empty()

    # Running the backtest function on the selected ticker
    run_backtest()

    if trade_button:
        high_confidence_trades = scan_for_trades(final_tickers, confidence_threshold)

        if high_confidence_trades:
            best_trade = sorted(high_confidence_trades, key=lambda x: x['confidence'], reverse=True)[0]
            send_email(f"High Confidence Trade: {best_trade['ticker']}", f"Trade Data: {best_trade['data']}")

            # Log the trade for future training
            log_trade({
                "ticker": best_trade['ticker'],
                "confidence": best_trade['confidence'],
                "data": best_trade['data'].to_dict()
            })
            st.success(f"✅ High-confidence trade found for {best_trade['ticker']}!")
        else:
            st.error("❌ No high-confidence trades found. Try again later.")

# Tickers for Backtesting and Scanning
final_tickers = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP", "QQQ", "SPY", "VIX"]

if __name__ == "__main__":
    main()


