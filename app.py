# ✅ SWING SNIPER GPT BOT (Updated with Fixes)

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import openai
import smtplib
from email.mime.text import MIMEText

# ✅ Email Settings (uses environment variables for safety)
import os
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# ✅ OpenAI Client
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Red Zone Panic Assets
panic_assets = ["IAU", "GLD", "SDS", "SH", "RWM", "VIXY", "UVXY", "TLT", "BIL", "SHY"]

# ✅ Main Asset Groups
tickers = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP",
    "WMT", "PEP", "KO", "CVS", "JNJ", "PG", "XOM", "VZ", "O",
    "JEPQ", "JEPI", "RYLD", "QYLD", "SCHD", "VYM", "DVY", "HDV",
    "QQQ", "SPY", "VOO", "IWM", "ARKK", "XLF", "XLE", "XLV", "XLC",
    "EWZ", "EEM", "FXI", "EWJ", "IBB",
    "BTC-USD", "ETH-USD"
]

# ✅ Streamlit UI
st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT Dashboard")
st.markdown("""
This AI-powered swing trading bot scans the market for high-confidence opportunities based on multiple indicators. You can control probability filters, Red Zone panic mode, and view AI trade insights in one click.
""")

# 🎛️ Sidebar Settings
confidence_threshold = st.sidebar.slider("🎯 Minimum Confidence (%)", 50, 95, 85, step=5)
red_zone = st.sidebar.checkbox("🛑 Activate Red Zone Mode")
start_button = st.sidebar.button("🚀 Start Scan")

# 🧠 GPT Recommendation Function
def get_gpt_recommendation(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GPT Error: {e}"

# 📧 Email Sender
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    except Exception as e:
        st.error(f"Email failed: {e}")

# 🔍 Scan Logic
if start_button:
    assets = panic_assets if red_zone else tickers
    st.markdown(f"### 🔴 Red Zone Mode: {'ON' if red_zone else 'OFF'}")
    high_confidence = []

    for ticker in assets:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = df.rename(columns=lambda x: x.capitalize())

            df = ta.add_all_ta_features(
                df,
                open="Open", high="High", low="Low", close="Close", volume="Volume",
                fillna=True
            )

            latest = df.iloc[-1]

            rsi = latest["momentum_rsi"]
            stochrsi = latest["momentum_stoch_rsi"]
            macd = latest["trend_macd"]
            macd_signal = latest["trend_macd_signal"]
            macd_hist = macd - macd_signal
            close = latest["Close"]
            ma20 = latest["trend_sma_fast"]
            ma150 = latest["trend_sma_slow"]
            bbm = latest["volatility_bbm"]
            bbw = latest["volatility_bbw"]
            bb_low = bbm - bbw
            volume = latest["Volume"]
            avg_volume = df["Volume"].values.ravel()[-20:].mean()
            adx = latest["trend_adx"]

            if (
                rsi < 35 and
                stochrsi < 0.2 and
                macd_hist > 0 and
                close > ma150 and close < ma20 and
                close <= bb_low and
                volume > avg_volume and
                adx > 20
            ):
                prompt = f"""
                You are an expert swing trader. Given the following data, write a 3-sentence recommendation:
                - Ticker: {ticker}
                - RSI: {rsi:.2f}
                - Stoch RSI: {stochrsi:.2f}
                - MACD Hist: {macd_hist:.3f}
                - Close: {close:.2f}
                - MA20: {ma20:.2f}, MA150: {ma150:.2f}
                - Volume: {int(volume)} vs Avg: {int(avg_volume)}
                - ADX: {adx:.2f}
                Include:
                - Confidence level from 0 to 100%
                - Entry, Target, Stop Loss
                - Reason for the trade
                """

                recommendation = get_gpt_recommendation(prompt)
                if recommendation:
                    st.success(f"📈 {ticker}")
                    st.write(recommendation)
                    send_email(f"🔔 Sniper Signal: {ticker}", recommendation)

        except Exception as e:
            st.warning(f"⚠️ {ticker} failed: {e}")

    st.balloons()
    st.info("Scan complete ✅")
""")


