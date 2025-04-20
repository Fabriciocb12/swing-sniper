import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# ------------------------ CONFIG ------------------------
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"
OPENAI_API_KEY = "your_openai_api_key"

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------ FUNCTIONS ------------------------
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        st.success("📧 Email sent successfully!")
    except Exception as e:
        st.error(f"⚠️ Email failed: {e}")

def run_analysis():
    tickers = st.session_state.tickers
    high_confidence_trades = []

    with st.spinner("🔍 Scanning tickers..."):
        for ticker in tickers:
            try:
                data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                data.rename(columns=lambda x: str(x).capitalize(), inplace=True)

                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    continue

                data = ta.add_all_ta_features(
                    data,
                    open="Open", high="High", low="Low", close="Close", volume="Volume",
                    fillna=True
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
                    high_confidence_trades.append({
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
                    })

            except Exception as e:
                st.warning(f"⚠️ {ticker} error: {e}")

    if not high_confidence_trades:
        st.error("❌ No high-confidence trades found today. Try again tomorrow.")
        return

    st.success("✅ High-confidence trade(s) found:")

    for trade in high_confidence_trades:
        st.subheader(f"📊 {trade['ticker']}")
        st.json(trade)

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
        - Confidence level from 0% to 100%
        - Suggested entry price (around close)
        - Suggested target price
        - Suggested stop loss
        - Why this trade is attractive
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            gpt_advice = response.choices[0].message.content
            st.markdown(f"**🧠 GPT Analysis:**\n\n{gpt_advice}")

            if st.button(f"📩 Send Email for {trade['ticker']}"):
                send_email(f"Sniper Signal: {trade['ticker']}", gpt_advice)

        except Exception as e:
            st.error(f"GPT Error: {e}")

# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="📈 Swing Sniper GPT", layout="centered")
st.title("🎯 Swing Sniper GPT Bot")
st.markdown("""
Welcome to **Swing Sniper GPT** — your AI-powered swing trade assistant built by Fabricio 💼

Click below to run a full scan across markets using AI + TA.
""")

if "tickers" not in st.session_state:
    st.session_state.tickers = [
        "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "SHOP",
        "WMT", "PEP", "KO", "CVS", "JNJ", "PG", "XOM", "VZ", "O",
        "JEPQ", "JEPI", "RYLD", "QYLD", "SCHD", "VYM", "DVY", "HDV",
        "QQQ", "SPY", "VOO", "IWM", "ARKK", "TLT", "XLF", "XLE", "XLV", "XLC",
        "EWZ", "EEM", "FXI", "EWJ", "IBB",
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "BNB-USD", "AVAX-USD",
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PALL"
    ]

if st.button("🚀 Run Full Scanner"):
    run_analysis()

st.markdown("""
---
👣 *Powered by ChatGPT · Crafted with 💡 by Fabricio Castro*
""")

