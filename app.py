# Swing Sniper GPT Bot - Final Unified Version
# This Streamlit app scans multiple stock tickers using technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
# to identify potential swing trade opportunities. It includes a confidence threshold filter, optional "Red Zone" mode,
# email notifications for alerts, a GPT-based analysis for the top trade signal, a backtesting tool in the sidebar,
# and logging of triggered trades to a local JSON file.

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import datetime
import matplotlib.pyplot as plt

# Optional: if you want to use OpenAI for GPT analysis, install and import openai
try:
    import openai
except ImportError:
    openai = None

# Optional: Email sending imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ========== Configuration ==========
# Email configuration (fill these with actual credentials if using email feature)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "youremail@example.com"    # replace with your sender email
EMAIL_PASSWORD = "yourpassword"            # replace with your email password or app-specific password
EMAIL_RECIPIENT = "recipient@example.com"  # replace with the recipient email for alerts

# Technical indicator calculation functions
def compute_RSI(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) for a given close price series."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Use simple moving average for RSI calculation
    avg_gain = up.rolling(window=period, min_periods=period).mean()
    avg_loss = down.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(close: pd.Series, fast=12, slow=26, signal=9):
    """Compute MACD line and signal line for given close price series."""
    # Exponential moving averages for MACD
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_Bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands for given close price series."""
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) for given high, low, close price series."""
    # True range = max(high-low, abs(high-prev_close), abs(low-prev_close))
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
    """Compute On-Balance Volume (OBV) for given close price and volume series."""
    obv = pd.Series(np.zeros_like(close), index=close.index)
    # OBV starts at 0 and accumulates volume on up days, subtracts on down days
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

# Pre-defined list of "panic" tickers for Red Zone mode (bear market indicators or inverse ETFs)
PANIC_TICKERS = ["^VIX", "UVXY", "SQQQ"]

# ========== User Interface (Streamlit) ==========
st.set_page_config(page_title="Swing Sniper GPT Bot", layout="wide")
st.title("Swing Sniper GPT Bot")

# Instructions/description (optional)
st.write("This bot scans selected stock tickers using RSI, MACD, Bollinger Bands, ATR, and OBV to identify potential swing trades.")
st.write("Adjust the confidence threshold to filter signals, enable Red Zone mode to include panic tickers, and use the backtesting tool in the sidebar.")

# Ticker input
tickers_input = st.text_input("Enter tickers to scan (comma-separated):", value="AAPL, MSFT, TSLA, GOOGL, AMZN")
# Parse tickers list
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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

# Backtesting feature in sidebar
st.sidebar.header("Backtesting")
backtest_ticker = st.sidebar.selectbox("Select Ticker for Backtest", options=tickers if tickers else ["AAPL"], index=0)
short_window = st.sidebar.number_input("Short SMA window", min_value=1, value=50)
long_window = st.sidebar.number_input("Long SMA window", min_value=1, value=200)
backtest_button = st.sidebar.button("Run Backtest")

if backtest_button:
    # Fetch historical data for backtest (using yfinance)
    import yfinance as yf
    try:
        data_bt = yf.download(backtest_ticker, period="1y", interval="1d")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch data for {backtest_ticker}: {e}")
        data_bt = None
    if data_bt is not None and not data_bt.empty:
        # Calculate SMA lines
        data_bt["SMA_short"] = data_bt["Close"].rolling(window=int(short_window)).mean()
        data_bt["SMA_long"] = data_bt["Close"].rolling(window=int(long_window)).mean()
        # Identify crossover points
        buy_signals = []
        sell_signals = []
        position = False  # False = no position, True = holding (long)
        for i in range(1, len(data_bt)):
            if not position:
                # look to buy
                if data_bt["SMA_short"].iloc[i] > data_bt["SMA_long"].iloc[i] and data_bt["SMA_short"].iloc[i-1] <= data_bt["SMA_long"].iloc[i-1]:
                    buy_signals.append((data_bt.index[i], data_bt["Close"].iloc[i]))
                    position = True
            else:
                # holding, look to sell
                if data_bt["SMA_short"].iloc[i] < data_bt["SMA_long"].iloc[i] and data_bt["SMA_short"].iloc[i-1] >= data_bt["SMA_long"].iloc[i-1]:
                    sell_signals.append((data_bt.index[i], data_bt["Close"].iloc[i]))
                    position = False
        # Plot the price and SMAs
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data_bt.index, data_bt["Close"], label="Close Price", color="black")
        ax.plot(data_bt.index, data_bt["SMA_short"], label=f"SMA {int(short_window)}", color="blue")
        ax.plot(data_bt.index, data_bt["SMA_long"], label=f"SMA {int(long_window)}", color="orange")
        # Plot buy/sell signals
        for date, price in buy_signals:
            ax.scatter(date, price, marker="^", color="green", label="Buy Signal")
        for date, price in sell_signals:
            ax.scatter(date, price, marker="v", color="red", label="Sell Signal")
        ax.set_title(f"{backtest_ticker} SMA Crossover Backtest")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.write("No data to backtest.")

# If Scan button is pressed, perform trade scanning
if scan_button:
    if not tickers:
        st.warning("Please enter at least one ticker symbol to scan.")
    else:
        # Combine base tickers with panic tickers if Red Zone mode is on
        tickers_to_scan = tickers.copy()
        if red_zone:
            # Add panic tickers (avoid duplicates)
            for pt in PANIC_TICKERS:
                if pt not in tickers_to_scan:
                    tickers_to_scan.append(pt)
        best_trade = None
        best_confidence = 0.0
        best_signals = None  # store indicator values for analysis
        # Import yfinance when needed for scanning
        import yfinance as yf
        for ticker in tickers_to_scan:
            try:
                data = yf.download(ticker, period="3mo", interval="1d")
            except Exception as e:
                st.write(f"Error fetching data for {ticker}: {e}")
                continue
            if data is None or data.empty:
                st.write(f"No data for {ticker}. Skipping.")
                continue
            # Compute indicators on the data
            close = data["Close"]
            high = data["High"]
            low = data["Low"]
            volume = data["Volume"]
            # Ensure enough data points for indicators
            if len(close) < 30:
                st.write(f"Not enough data for {ticker} to compute indicators. Skipping.")
                continue
            rsi_series = compute_RSI(close)
            rsi_value = rsi_series.iloc[-1]
            macd_line, signal_line, macd_hist = compute_MACD(close)
            macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
            # Check if MACD had a bullish crossover recently
            macd_bull = False
            if len(macd_line) >= 2:
                prev_macd_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
                if macd_diff > 0 and prev_macd_diff < 0:
                    macd_bull = True
            # Bollinger Bands
            ma20, upper_band, lower_band = compute_Bollinger(close)
            last_close = close.iloc[-1]
            upper_band_last = upper_band.iloc[-1]
            lower_band_last = lower_band.iloc[-1]
            # ATR
            atr_series = compute_ATR(high, low, close)
            atr_value = atr_series.iloc[-1]
            atr_percent = (atr_value / last_close) * 100 if last_close != 0 else 0
            # OBV
            obv_series = compute_OBV(close, volume)
            obv_now = obv_series.iloc[-1]
            obv_past = obv_series.iloc[-6] if len(obv_series) >= 6 else obv_series.iloc[0]
            obv_trend_up = obv_now > obv_past
            # Determine bullish signals (1 or 0 for each)
            signals_count = 0
            total_signals = 4  # counting RSI, MACD, Bollinger, OBV for bullishness
            # RSI signal
            if rsi_value < 30:  # oversold
                signals_count += 1
            # MACD signal
            if macd_diff > 0:
                # If just crossed, or currently bullish
                signals_count += 1
            # Bollinger signal
            if last_close < lower_band_last:
                signals_count += 1
            # OBV signal
            if obv_trend_up:
                signals_count += 1
            # Confidence base is ratio of bullish signals
            if total_signals > 0:
                conf_base = (signals_count / total_signals) * 100.0
            else:
                conf_base = 0.0
            # Apply ATR penalty if volatility is high
            atr_penalty_factor = 1.0
            if atr_percent > 10:
                atr_penalty_factor = 0.8  # reduce confidence by 20% if ATR > 10% of price
            elif atr_percent > 5:
                atr_penalty_factor = 0.9  # reduce by 10% if ATR between 5-10%
            confidence = conf_base * atr_penalty_factor
            # Check if this trade meets threshold and is highest so far
            if confidence >= threshold and confidence > best_confidence:
                best_confidence = confidence
                best_trade = ticker
                # store values for analysis of best trade
                best_signals = {
                    "RSI": rsi_value,
                    "MACD_diff": macd_diff,
                    "MACD_bull_cross": macd_bull,
                    "Close": last_close,
                    "UpperBand": upper_band_last,
                    "LowerBand": lower_band_last,
                    "ATR%": atr_percent,
                    "OBV_trend_up": obv_trend_up
                }
        # After scanning all tickers
        if best_trade:
            # Display alert for best trade
            alert_placeholder.success(f"Trade Alert: **{best_trade}** with confidence **{best_confidence:.1f}%** (Threshold {threshold}%)")
            # Generate analysis text for the best trade
            analysis_text = ""
            if best_signals:
                # Option 1: Use OpenAI API if available and API key is set
                api_key = None
                if openai:
                    # Check if API key is provided in environment or Streamlit secrets
                    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or None
                    try:
                        api_key = api_key or st.secrets["OPENAI_API_KEY"]
                    except Exception:
                        pass
                if openai and api_key:
                    openai.api_key = api_key
                    prompt = (f"Stock {best_trade} Technical Analysis:\n"
                              f"RSI: {best_signals['RSI']:.1f}\n"
                              f"MACD Diff: {best_signals['MACD_diff']:.4f}{' (bullish crossover)' if best_signals['MACD_bull_cross'] else ''}\n"
                              f"Last Close: {best_signals['Close']:.2f}, Bollinger Lower: {best_signals['LowerBand']:.2f}, Upper: {best_signals['UpperBand']:.2f}\n"
                              f"ATR%: {best_signals['ATR%']:.2f}%\n"
                              f"OBV trend up: {best_signals['OBV_trend_up']}\n"
                              "Provide a brief analysis of these indicators and the stock's outlook.")
                    try:
                        response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=prompt,
                            max_tokens=150,
                            n=1,
                            stop=None,
                            temperature=0.5
                        )
                        analysis_text = response.choices[0].text.strip()
                    except Exception as e:
                        analysis_text = ""  # if API call fails, fall back to rule-based
                # Option 2: Rule-based analysis (if no API or as fallback)
                if analysis_text == "":
                    rsi_val = best_signals["RSI"]
                    macd_val = best_signals["MACD_diff"]
                    macd_cross = best_signals["MACD_bull_cross"]
                    last_close = best_signals["Close"]
                    lower_band_val = best_signals["LowerBand"]
                    upper_band_val = best_signals["UpperBand"]
                    atr_perc = best_signals["ATR%"]
                    obv_up = best_signals["OBV_trend_up"]
                    # Construct a simple analysis based on indicator values
                    if rsi_val < 30:
                        analysis_text += f"RSI is {rsi_val:.1f}, indicating the stock is oversold. "
                    elif rsi_val > 70:
                        analysis_text += f"RSI is {rsi_val:.1f}, indicating overbought conditions. "
                    else:
                        analysis_text += f"RSI is {rsi_val:.1f}, which is in a neutral range. "
                    if macd_cross:
                        analysis_text += "MACD has just made a bullish crossover, suggesting upward momentum. "
                    elif macd_val > 0:
                        analysis_text += "MACD is above the signal line, confirming a bullish trend. "
                    else:
                        analysis_text += "MACD is below the signal line, showing no bullish momentum yet. "
                    if last_close < lower_band_val:
                        analysis_text += "Price is trading below the lower Bollinger Band, which often indicates an oversold reversal opportunity. "
                    elif last_close > upper_band_val:
                        analysis_text += "Price is above the upper Bollinger Band, indicating overbought conditions. "
                    else:
                        analysis_text += "Price is within Bollinger Bands normal range. "
                    if obv_up:
                        analysis_text += "On-balance volume is rising, indicating buying pressure. "
                    else:
                        analysis_text += "On-balance volume is flat or falling, not confirming an increase in buying interest. "
                    analysis_text += f"Overall confidence in this trade is {best_confidence:.1f}% based on current indicators."
            # Display the analysis
            analysis_placeholder.markdown("**GPT Analysis:** " + analysis_text)
            # Send email notification
            try:
                if EMAIL_ADDRESS != "youremail@example.com":  # only attempt if credentials are likely set
                    msg = MIMEMultipart()
                    msg["From"] = EMAIL_ADDRESS
                    msg["To"] = EMAIL_RECIPIENT
                    msg["Subject"] = f"Trade Alert: {best_trade} at {best_confidence:.1f}% confidence"
                    body = (f"Swing Sniper GPT Bot has detected a trade opportunity for {best_trade}.\n"
                            f"Confidence: {best_confidence:.1f}% (Threshold set at {threshold}%).\n"
                            f"Indicators summary:\n"
                            f" - RSI: {best_signals['RSI']:.1f}\n"
                            f" - MACD diff: {best_signals['MACD_diff']:.4f}{' (bullish crossover)' if best_signals['MACD_bull_cross'] else ''}\n"
                            f" - Last Close vs Bollinger Bands: {last_close:.2f} (L: {best_signals['LowerBand']:.2f}, U: {best_signals['UpperBand']:.2f})\n"
                            f" - ATR%: {best_signals['ATR%']:.2f}%\n"
                            f" - OBV trend up: {best_signals['OBV_trend_up']}\n\n"
                            f"GPT Analysis:\n{analysis_text}\n")
                    msg.attach(MIMEText(body, "plain"))
                    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
                    server.starttls()
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.send_message(msg)
                    server.quit()
                    st.write("Email notification sent.")
                else:
                    st.info("Email alert not sent (email credentials not configured).")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
            # Log the trade to a local JSON file
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": best_trade,
                "confidence": round(best_confidence, 2),
                "threshold": threshold,
                "indicators": best_signals
            }
            try:
                log_file = "trade_logs.json"
                # Load existing log file if exists
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        logs = json.load(f)
                else:
                    logs = []
                logs.append(log_entry)
                with open(log_file, "w") as f:
                    json.dump(logs, f, indent=4)
            except Exception as e:
                st.error(f"Failed to log trade: {e}")
        else:
            # No trade met the threshold
            alert_placeholder.warning(f"No trade signals exceeded the confidence threshold of {threshold}%.")
            analysis_placeholder.empty()
