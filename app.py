# ✅ app.py: Fully upgraded with model integration, JSON state tracking, and email alerts

import os
import json
import joblib
import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI

# ✅ Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# ✅ Load trained models
entry_model = joblib.load("swing_sniper_model.pkl")
exit_model = joblib.load("swing_sniper_exit_model_v2.pkl")

# ✅ Load lock file
LOCK_FILE = "locked_positions.json"
if not os.path.exists(LOCK_FILE):
    with open(LOCK_FILE, "w") as f:
        json.dump({"positions": []}, f)

def load_locks():
    with open(LOCK_FILE, "r") as f:
        return json.load(f)

def save_locks(state):
    with open(LOCK_FILE, "w") as f:
        json.dump(state, f, indent=2)

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Streamlit UI
st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT")

confidence_threshold = st.slider("Set Trade Confidence Threshold %", 50, 95, 80, step=10)
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

# ✅ Main Ticker List
main_tickers = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP",
    "WMT", "PEP", "KO", "CVS", "JNJ", "PG", "XOM", "VZ", "O",
    "JEPQ", "JEPI", "RYLD", "QYLD", "SCHD", "VYM", "DVY", "HDV",
    "QQQ", "SPY", "VOO", "IWM", "ARKK", "XLF", "XLE", "XLV", "XLC",
    "EWZ", "EEM", "FXI", "EWJ", "IBB",
    "BTC-USD", "ETH-USD"
]

final_tickers = main_tickers + panic_assets
lock_state = load_locks()

if trade_button:
    active_trades = []
    for ticker in final_tickers:
        try:
            status_placeholder.info(f"📊 Scanning {ticker}...")
            data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
            if data is None or data.empty:
                continue

            df = data.copy()
            df = df.dropna()
            df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["Close"]).stochrsi()
            df["hma"] = df["Close"].rolling(window=21).mean()
            df["hma_breakout"] = df["Close"] > df["hma"]
            df["smo"] = df["Close"] - ta.volatility.BollingerBands(df["Close"]).bollinger_mavg()

            latest = df.iloc[-1]
            features = pd.DataFrame([{
                "rsi": latest["rsi"],
                "stoch_rsi": latest["stoch_rsi"],
                "hma_breakout": int(latest["hma_breakout"]),
                "smo": latest["smo"]
            }])

            prediction = entry_model.predict(features)[0]

            if prediction == 1 and ticker not in lock_state["positions"]:
                # ✅ Entry Signal
                st.success(f"✅ Entry Signal: {ticker}")
                lock_state["positions"].append(ticker)
                save_locks(lock_state)

                send_email(f"🔔 Entry Signal: {ticker}", f"Swing Sniper GPT recommends entering {ticker} today.")
            elif ticker in lock_state["positions"]:
                # ✅ Exit Model Check
                exit_prediction = exit_model.predict(features)[0]
                if exit_prediction == 1:
                    st.warning(f"🔻 Exit Signal: {ticker}")
                    lock_state["positions"].remove(ticker)
                    save_locks(lock_state)

                    send_email(f"🔻 Exit Signal: {ticker}", f"Swing Sniper GPT recommends exiting {ticker} today.")

        except Exception as e:
            st.error(f"⚠️ Error with {ticker}: {e}")

    if not lock_state["positions"]:
        st.info("ℹ️ No active swing trades in lock.")
    else:
        st.markdown("### 🔒 Currently Locked Positions:")
        st.write(lock_state["positions"])

# ✅ GPT Sector Analysis
with st.expander("🧠 Sector Sentiment Snapshot"):
    try:
        gpt_prompt = """
        Act as a market strategist. Analyze macro trends, earnings, seasonality, and technicals.
        Briefly rate each sector: + (bullish), - (bearish), ~ (neutral).
        Summarize top 3 opportunities.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": gpt_prompt}]
        )
        summary = response.choices[0].message.content
        st.text_area("GPT Sector Analysis", value=summary, height=300)
        send_email("📊 Sector Sentiment Snapshot", summary)
    except Exception as e:
        st.error(f"❌ GPT Error: {e}")
