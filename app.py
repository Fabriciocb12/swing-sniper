# Full app.py - Swing Sniper GPT Bot
# This bot scans multiple tickers, calculates technical indicators, filters based on confidence,
# sends email notifications for high-confidence trades, and logs the trades for future learning.

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import yfinance as yf

# Optional: If you have OpenAI GPT setup
try:
    import openai
except ImportError:
    openai = None

# Technical Indicators Calculation Functions
def compute_RSI(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=period, min_periods=period).mean()
    avg_loss = down.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(close: pd.Series, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_Bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - prev_close).abs(),
        'lc': (low - prev_close).abs()
    })
    tr = tr.max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def compute_OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = pd.Series(np.zeros_like(close), index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

# Email configuration (replace with your actual details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "youremail@example.com"  # Replace with your email
EMAIL_PASSWORD = "yourpassword"  # Replace with your email password
EMAIL_RECIPIENT = "recipient@example.com"  # Replace with recipient's email

# ========== User Interface (Streamlit) ==========
st.set_page_config(page_title="Swing Sniper GPT Bot", layout="wide")
st.title("Swing Sniper GPT Bot")

# Ticker input
tickers_input = st.text_input("Enter tickers to scan (comma-separated):", value="AAPL, MSFT, TSLA, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Red Zone mode toggle
red_zone = st.checkbox("Red Zone Mode (include panic tickers)")

# Confidence threshold slider (1% to 100%)
threshold = st.slider("Confidence Threshold (%)", min_value=1, max_value=100, value=50, step=1)
st.write(f"Current sensitivity threshold: {threshold}%")

# Scan button
scan_button = st.button("Scan for Trades")

# Placeholder for scan result display
alert_placeholder = st.empty()
analysis_placeholder = st.empty()

# Fetch data function (yfinance)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d")
        return data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

# Scan for trades function
def scan_for_trades(tickers, threshold):
    best_trade = None
    best_confidence = 0.0
    best_signals = None  # Store indicator values for analysis of best trade
    for ticker in tickers:
        data = fetch_data(ticker)
        if data is None or data.empty:
            continue

        close = data['Close']
        rsi = compute_RSI(close).iloc[-1]
        macd_line, signal_line, macd_hist = compute_MACD(close)
        ma, upper_band, lower_band = compute_Bollinger(close)
        atr = compute_ATR(data['High'], data['Low'], close).iloc[-1]
        obv = compute_OBV(close, data['Volume']).iloc[-1]

        # Confidence calculation based on indicators
        conf_base = 0
        if rsi < 30:
            conf_base += 25  # RSI oversold
        if macd_hist > 0:
            conf_base += 25  # MACD bullish
        if close.iloc[-1] < lower_band.iloc[-1]:
            conf_base += 25  # Bollinger Band oversold
        if obv > obv.shift(1).iloc[-1]:
            conf_base += 25  # OBV upward

        confidence = conf_base * (1 - atr / 100)  # Adjust confidence based on ATR (volatility)
        
        if confidence > threshold and confidence > best_confidence:
            best_confidence = confidence
            best_trade = ticker
            best_signals = {
                "RSI": rsi,
                "MACD_hist": macd_hist,
                "Close": close.iloc[-1],
                "UpperBand": upper_band.iloc[-1],
                "LowerBand": lower_band.iloc[-1],
                "ATR": atr,
                "OBV": obv
            }

    return best_trade, best_confidence, best_signals

# If Scan button is pressed, perform trade scanning
if scan_button:
    best_trade, best_confidence, best_signals = scan_for_trades(tickers, threshold)
    if best_trade:
        # Display alert for best trade
        alert_placeholder.success(f"Trade Alert: **{best_trade}** with confidence **{best_confidence:.1f}%**")
        # Generate analysis text for the best trade
        analysis_text = f"RSI: {best_signals['RSI']}, MACD: {best_signals['MACD_hist']}, ATR: {best_signals['ATR']}, OBV: {best_signals['OBV']}"
        analysis_placeholder.markdown("**Analysis**: " + analysis_text)
        
        # Send email notification
        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = EMAIL_RECIPIENT
            msg["Subject"] = f"Trade Alert: {best_trade} at {best_confidence:.1f}% confidence"
            body = f"Confidence: {best_confidence:.1f}%\n\n{analysis_text}"
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            st.write("Email notification sent.")
        except Exception as e:
            st.error(f"Failed to send email: {e}")

        # Log the trade to a local JSON file for training later
        trade_log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": best_trade,
            "confidence": round(best_confidence, 2),
            "signals": best_signals
        }
        log_file = "trade_logs.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
            logs.append(trade_log_entry)
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            st.error(f"Failed to log trade: {e}")
    else:
        alert_placeholder.warning(f"No trades met the confidence threshold of {threshold}%.")
