import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import backtest functions from backtest_tab.py
from backtest_tab import backtest_tab

# Load from secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT")

# Adjust the confidence threshold slider to include 30% and 40%
confidence_threshold = st.slider("Set Trade Confidence Threshold %", 30, 95, 80, step=10)

red_zone_enabled = st.checkbox("🚨 Enable Red Zone Mode (Bear Market Protocol)")

if red_zone_enabled:
    st.markdown("### 🔴 Red Zone Mode: ON")
    panic_assets = ["IAU", "GLD", "SDS", "SH", "RWM", "VIXY", "UVXY", "TLT", "BIL", "SHY"]
else:
    st.markdown("### 🟢 Red Zone Mode: OFF")
    panic_assets = []

trade_button = st.button("🔍 Scan for Trades")
status_placeholder = st.empty()

# ✅ Email Sender
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

# ✅ Load Locked Positions from JSON
LOCK_FILE = "locked_positions.json"
if os.path.exists(LOCK_FILE):
    with open(LOCK_FILE, "r") as f:
        locked_positions = json.load(f)
else:
    locked_positions = {}

# ✅ Save Locked Positions to JSON
def save_locked_positions():
    with open(LOCK_FILE, "w") as f:
        json.dump(locked_positions, f, indent=2)

# Tickers
main_tickers = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP",
    "WMT", "PEP", "KO", "CVS", "JNJ", "PG", "XOM", "VZ", "O",
    "JEPQ", "JEPI", "RYLD", "QYLD", "SCHD", "VYM", "DVY", "HDV",
    "QQQ", "SPY", "VOO", "IWM", "ARKK", "XLF", "XLE", "XLV", "XLC",
    "EWZ", "EEM", "FXI", "EWJ", "IBB",
    "BTC-USD", "ETH-USD"
]

final_tickers = main_tickers + panic_assets

# 🧠 Backtest Functionality
# Add a new section for the backtest tab
st.sidebar.title("Backtest Strategy")
selected_ticker_backtest = st.sidebar.selectbox("Select Ticker for Backtest", final_tickers)

if st.sidebar.button("Run Backtest"):
    st.write(f"Running backtest for {selected_ticker_backtest}...")

    # ✅ Fetch Data for the selected ticker
    def fetch_data(ticker):
        data = yf.download(ticker, period="1y", interval="1d")
        return data

    data = fetch_data(selected_ticker_backtest)

    # ✅ Backtest Strategy (simple moving average crossover)
    backtest_results = backtest_tab.backtest_strategy(data)  # Using the backtest function from backtest_tab.py

    # ✅ Plot Results
    backtest_tab.plot_results(backtest_results)  # Plot the results using the plot function from backtest_tab.py

# Main trade scan button (already existing)
if trade_button:
    high_confidence_trades = []
    for ticker in final_tickers:
        try:
            if ticker in locked_positions:
                st.warning(f"🔒 {ticker} is already locked-in.")
                continue

            status_placeholder.info(f"🔎 Scanning {ticker}...")

            data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)

            # Flatten the dataframe if needed (remove multi-index columns)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]  # Flatten to single level

            # Manually calculate RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # Debugging: Ensure RSI is being added to DataFrame
            st.write(data[['Close', 'RSI']].tail())  # Display last few rows of Close and RSI

            if 'RSI' not in data.columns:
                st.warning(f"RSI calculation failed for {ticker}")
                continue

            # Calculate other indicators as before
            data['bb_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['bb_lower'] = ta.volatility.bollinger_lband(data['Close'])
            data['bb_width'] = data['bb_upper'] - data['bb_lower']  # Manual width calculation

            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            data['MACD'] = ta.trend.macd(data['Close'])
            data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            data['EMA9'] = ta.trend.ema_indicator(data['Close'], window=9)
            data['EMA21'] = ta.trend.ema_indicator(data['Close'], window=21)

            # Get dynamic confidence level based on new criteria
            confidence_threshold = get_trade_confidence(data)

            # Evaluate trade criteria (loosened conditions for 30-40% confidence range)
            match = False
            if (
                data['RSI'].iloc[-1] < 50 and
                data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] and
                data['Close'].iloc[-1] > data['bb_upper'].iloc[-1] and
                data['ATR'].iloc[-1] > data['ATR'].mean() and
                data['OBV'].iloc[-1] > data['OBV'].iloc[-2]
            ):
                match = True

            if match:
                trade = {
                    "ticker": ticker,
                    "rsi": round(data['RSI'].iloc[-1], 2),
                    "macd_hist": round(data['MACD_hist'].iloc[-1], 3),
                    "close": round(data['Close'].iloc[-1], 2),
                    "ema9": round(data['EMA9'].iloc[-1], 2),
                    "ema21": round(data['EMA21'].iloc[-1], 2),
                    "volume": round(data['Volume'].iloc[-1]),
                    "bb_upper": round(data['bb_upper'].iloc[-1], 2),
                }

                # Send email and log trade details
                prompt = f"""
                You are an expert swing trader. Given the following data, write a 3-4 sentence recommendation:
                - Ticker: {trade['ticker']}
                - RSI: {trade['rsi']}
                - MACD Histogram: {trade['macd_hist']}
                - Close: {trade['close']}
                - EMA9: {trade['ema9']}
                - EMA21: {trade['ema21']}
                - Volume: {trade['volume']}
                - Bollinger Band Upper: {trade['bb_upper']}

                Include:
                - Confidence level from 0% to 100%
                - Suggested entry price (around close)
                - Suggested target price
                - Suggested stop loss
                - Why this trade is attractive
                """

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                gpt_output = response.choices[0].message.content
                st.markdown(f"### 📈 {trade['ticker']}")
                st.text(gpt_output)
                high_confidence_trades.append(trade)
                send_email(f"🔔 SniperBot Signal: {trade['ticker']}", gpt_output)

                # ✅ Lock the trade
                locked_positions[ticker] = {
                    "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": float(trade['close']),
                    "note": "Locked by model"
                }
                save_locked_positions()

        except Exception as e:
            st.warning(f"⚠️ {ticker} failed: {e}")

    if not high_confidence_trades:
        st.error("❌ No high-confidence trades found. Try again later.")
    else:
        st.success("✅ High-confidence trade(s) found!")
