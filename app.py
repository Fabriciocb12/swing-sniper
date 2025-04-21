# ✅ SWING SNIPER GPT BOT - Phase 3.4 (Red Zone, HMA, Squeeze, Slider, UI Boost)
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from openai import OpenAI
import os
import smtplib
from email.mime.text import MIMEText

# ✅ ENV variables (Set these in Render dashboard under Environment section)
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Main UI
st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT")
st.markdown("""
<style>
.big-banner {font-size: 1.3rem; background: #111; color: lime; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ✅ Slider: Success Probability Threshold
confidence_threshold = st.slider("Minimum Confidence %", 50, 95, 80, step=10)

# ✅ Red Zone Protocol Toggle
red_zone = st.toggle("🚨 Red Zone Protocol")

# ✅ Button to Launch Scan
if st.button("🔍 Scan Market"):
    st.markdown('<div class="big-banner">Scanning tickers... stand by 🔄</div>', unsafe_allow_html=True)

    panic_assets = ["IAU", "GLD", "SDS", "SH", "RWM", "VIXY", "UVXY", "TLT", "BIL", "SHY", "BTC-USD", "ETH-USD"]
    default_tickers = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "JEPQ", "JEPI", "QQQ", "SPY"]
    tickers = panic_assets if red_zone else default_tickers

    high_confidence_trades = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
            data.rename(columns=lambda x: str(x).capitalize(), inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                continue

            # TA Features
            data = ta.add_all_ta_features(
                data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )

            # HMA Breakout (custom)
            data['HMA'] = data['Close'].rolling(window=20).mean().rolling(window=2).mean()
            breakout = data['Close'].iloc[-1] > data['HMA'].iloc[-1]

            # Squeeze Momentum Oscillator (simplified)
            data['squeeze'] = data['volatility_bbw'] < data['volatility_bbw'].rolling(window=20).mean()
            squeeze_trigger = data['squeeze'].iloc[-1] == False

            latest = data.iloc[-1]
            rsi = latest["momentum_rsi"]
            stochrsi = latest["momentum_stoch_rsi"]
            macd_hist = latest["trend_macd"] - latest["trend_macd_signal"]
            adx = latest["trend_adx"]
            volume = latest["Volume"]
            avg_volume = data["Volume"].rolling(window=20).mean().iloc[-1]
            close = latest["Close"]

            # Criteria
            if (
                rsi < 35 and stochrsi < 0.2 and macd_hist > 0 and adx > 20 and
                volume > avg_volume and breakout and squeeze_trigger
            ):
                # GPT Recommendation
                prompt = f"""
                You're an expert swing trader. Based on:
                RSI: {rsi}, StochRSI: {stochrsi}, MACD-Hist: {macd_hist}, ADX: {adx}, Volume Spike
                Also includes HMA breakout and Squeeze Momentum confirmation.
                Give recommendation with confidence level from 50-95%. Return entry, target, and stop loss.
                """

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                # Email Alert
                msg = MIMEText(response.choices[0].message.content)
                msg["Subject"] = f"📈 SniperBot Triggered: {ticker}"
                msg["From"] = EMAIL_SENDER
                msg["To"] = EMAIL_RECEIVER
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                    server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

                # Store
                high_confidence_trades.append((ticker, response.choices[0].message.content))

        except Exception as e:
            st.error(f"⚠️ {ticker} failed: {e}")

    if not high_confidence_trades:
        st.warning("No trades hit the target today.")
    else:
        for ticker, recommendation in high_confidence_trades:
            st.success(f"📊 {ticker}")
            st.markdown(recommendation)

