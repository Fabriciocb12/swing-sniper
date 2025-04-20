import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# 🌟 Email Configuration
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        st.success("📬 Email sent.")
    except Exception as e:
        st.error(f"⚠️ Email failed: {e}")

# 🔐 OpenAI Key
client = OpenAI(api_key="your_openai_api_key")

st.set_page_config(page_title="Swing Sniper GPT", layout="wide")
st.title("🎯 Swing Sniper GPT")

with st.sidebar:
    st.header("🔍 Scan Settings")
    selected_tickers = st.text_area("Tickers (comma-separated)", "AAPL,TSLA,NVDA,AMD,MSFT")
    run_scan = st.button("🚀 Run Scan")

if run_scan:
    tickers = [t.strip().upper() for t in selected_tickers.split(",")]
    high_confidence_trades = []

    for ticker in tickers:
        try:
            st.write(f"🔎 Scanning **{ticker}**...")
            data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            data.rename(columns=lambda x: str(x).capitalize(), inplace=True)
            data = ta.add_all_ta_features(
                data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )

            latest = data.iloc[-1]
            rsi = latest["momentum_rsi"]
            stochrsi = latest["momentum_stoch_rsi"]
            macd = latest["trend_macd"]
            macd_signal = latest["trend_macd_signal"]
            macd_hist = macd - macd_signal
            ma20 = latest["trend_sma_fast"]
            ma150 = latest["trend_sma_slow"]
            close = latest["Close"]
            bbm = latest["volatility_bbm"]
            bbw = latest["volatility_bbw"]
            bb_low = bbm - bbw
            volume = latest["Volume"]
            avg_volume = data["Volume"].rolling(window=20).mean().iloc[-1]
            adx = latest["trend_adx"]

            if (
                rsi < 35 and
                stochrsi < 0.2 and
                macd > macd_signal and
                macd_hist > 0 and
                close > ma150 and close < ma20 and
                close <= bb_low and
                volume > avg_volume and
                adx > 20
            ):
                trade = {
                    "ticker": ticker,
                    "rsi": round(rsi, 2),
                    "stochrsi": round(stochrsi, 2),
                    "macd_hist": round(macd_hist, 3),
                    "close": round(close, 2),
                    "ma20": round(ma20, 2),
                    "ma150": round(ma150, 2),
                    "volume": round(volume),
                    "avg_volume": round(avg_volume),
                    "adx": round(adx, 2)
                }

                prompt = f"""
                You are an expert swing trader. Given the following data, write a 3-4 sentence recommendation:
                - Ticker: {trade['ticker']}
                - RSI: {trade['rsi']}
                - Stochastic RSI: {trade['stochrsi']}
                - MACD Histogram: {trade['macd_hist']}
                - Close: {trade['close']}
                - MA20: {trade['ma20']}
                - MA150: {trade['ma150']}
                - Volume: {trade['volume']} vs avg {trade['avg_volume']}
                - ADX: {trade['adx']}

                Include:
                - Confidence level (0–100%)
                - Entry price (around close)
                - Target price
                - Stop loss
                - Why this trade is attractive
                """

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                recommendation = response.choices[0].message.content
                st.subheader(f"📈 {ticker} Setup")
                st.code(recommendation)
                send_email(f"🔔 SniperBot Signal: {ticker}", f"📊 {ticker}\n{recommendation}")
        except Exception as e:
            st.error(f"⚠️ Error scanning {ticker}: {e}")
