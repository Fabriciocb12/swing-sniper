import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os

# --- ENV VARS (set these in Render dashboard for security) ---
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- EMAIL FUNCTION ---
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ Email sent.")
    except Exception as e:
        print("⚠️ Email failed:", e)

# --- UI LAYOUT ---
st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT")

st.sidebar.header("🔧 Scanner Settings")
confidence_threshold = st.sidebar.slider("Min Confidence %", 50, 95, 80, 5)
run_button = st.sidebar.button("🔍 Run Scanner")

# --- WATCHLIST ---
tickers = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP",
    "WMT", "PEP", "KO", "CVS", "JNJ", "PG", "XOM", "VZ", "O",
    "JEPQ", "JEPI", "RYLD", "QYLD", "SCHD", "VYM", "DVY", "HDV",
    "QQQ", "SPY", "VOO", "IWM", "ARKK", "TLT", "XLF", "XLE", "XLV", "XLC",
    "EWZ", "EEM", "FXI", "EWJ", "IBB",
    "BTC-USD", "ETH-USD",
    # 🛑 Red Zone Assets
    "IAU", "GLD", "SDS", "SH", "RWM", "VIXY", "UVXY", "BIL", "SHY"
]

# --- RED ZONE PROTOCOL ---
def is_red_zone():
    try:
        spy = yf.download("SPY", period="5d", interval="1d", auto_adjust=True)["Close"]
        vix = yf.download("^VIX", period="5d", interval="1d", auto_adjust=True)["Close"]
        return (spy.pct_change().iloc[-1] < -0.015) and (vix.iloc[-1] > 20)
    except:
        return False

# --- SCANNER CORE ---
def scan_ticker(ticker, threshold):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
        if df.empty or df.shape[0] < 20:
            return None

        df.rename(columns=str.capitalize, inplace=True)
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        # Fix 2D array indicators
        df['hma'] = ta.trend.hull_moving_average(df['Close'], window=20).squeeze()

        sma20 = df['Close'].rolling(window=20).mean().squeeze()
        std = df['Close'].rolling(window=20).std().squeeze()
        upper = sma20 + 2 * std
        lower = sma20 - 2 * std
        df['squeeze_on'] = (df['Close'] > lower) & (df['Close'] < upper)

        latest = df.iloc[-1]
        volume_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
        macd_hist = latest['trend_macd'] - latest['trend_macd_signal']

        if (
            latest['momentum_rsi'] < 35 and
            latest['momentum_stoch_rsi'] < 0.2 and
            macd_hist > 0 and
            latest['Close'] > latest['trend_sma_slow'] and latest['Close'] < latest['trend_sma_fast'] and
            latest['Volume'] > volume_avg and
            latest['trend_adx'] > 20 and
            latest['squeeze_on'] and
            latest['Close'] > latest['hma']
        ):
            # GPT RECOMMENDATION
            prompt = f"""
            You are an expert swing trader. Given the following data, write a 4-sentence recommendation:
            Ticker: {ticker}
            RSI: {latest['momentum_rsi']:.2f}
            Stoch RSI: {latest['momentum_stoch_rsi']:.2f}
            MACD Hist: {macd_hist:.3f}
            Close: {latest['Close']:.2f}
            HMA: {latest['hma']:.2f}
            MA20: {latest['trend_sma_fast']:.2f}
            MA150: {latest['trend_sma_slow']:.2f}
            Volume: {latest['Volume']:.0f} vs avg {volume_avg:.0f}
            ADX: {latest['trend_adx']:.2f}

            Include:
            - Confidence % (realistic)
            - Entry/Exit suggestion
            - Why this trade looks strong
            """
            gpt = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content

            conf_line = [line for line in gpt.split("\n") if "%" in line]
            if conf_line:
                conf_num = int(''.join(filter(str.isdigit, conf_line[0])))
                if conf_num < threshold:
                    return None

            return {"ticker": ticker, "summary": gpt, "confidence": conf_num}
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ {ticker} failed: {e}")
        return None

# --- MAIN APP ---
if run_button:
    st.info("Running Swing Sniper... please wait ~20 sec")
    red_zone = is_red_zone()
    st.markdown(f"### 🔴 Red Zone Mode: {'ON' if red_zone else 'OFF'}")

    trades = []
    for t in tickers:
        if not red_zone and t in ["SDS", "SH", "RWM", "VIXY", "UVXY", "IAU", "GLD", "TLT", "BIL", "SHY"]:
            continue
        result = scan_ticker(t, confidence_threshold)
        if result:
            trades.append(result)

    if trades:
        st.success("🚀 Trade opportunities found!")
        for t in trades:
            with st.expander(f"📊 {t['ticker']} ({t['confidence']}%)"):
                st.markdown(t['summary'])
                send_email(f"🔔 SwingSniper Alert: {t['ticker']}", t['summary'])
    else:
        st.error("❌ No high-confidence trades found today. Try again later.")

# --- FOOTER ---
st.markdown("""
---
🔧 Built by Fabricio ⚡ Powered by GPT-4 | v3.4
""")


