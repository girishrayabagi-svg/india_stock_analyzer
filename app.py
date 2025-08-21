import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📈 Indian Stock Analyzer")
stock_list = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS"
]
ticker = st.sidebar.selectbox("Choose a Stock", stock_list, index=0)
years = st.sidebar.slider("Years of history", 1, 15, 5)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# -------------------------------
# Data Fetching
# -------------------------------
end = datetime.datetime.today()
start = end - datetime.timedelta(days=365*years)
data = yf.download(ticker, start=start, end=end, interval=interval)

if data.empty:
    st.error("⚠️ No data found for this stock.")
    st.stop()

st.title(f"📊 Stock Analysis: {ticker}")

# -------------------------------
# Price Chart
# -------------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Candlestick"
))
fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig)

# -------------------------------
# Moving Averages
# -------------------------------
st.subheader("Moving Averages")
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

plt.figure(figsize=(10,5))
plt.plot(data['Close'], label="Close Price")
plt.plot(data['MA20'], label="20-day MA")
plt.plot(data['MA50'], label="50-day MA")
plt.legend()
st.pyplot(plt)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔮 Stock Price Prediction (Linear Regression)")

try:
    df = data[['Close']].dropna().reset_index()
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    X = df[['Days']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write(f"✅ Model RMSE: {rmse:.2f}")

    # Future prediction (next 30 days)
    future_days = 30
    last_day = df['Days'].max()
    future_X = pd.DataFrame({'Days': np.arange(last_day+1, last_day+future_days+1)})
    future_pred = model.predict(future_X)

    plt.figure(figsize=(10,5))
    plt.plot(df['Days'], df['Close'], label="Historical")
    plt.plot(X_test, y_pred, label="Test Prediction")
    plt.plot(future_X['Days'], future_pred, label="Future Prediction")
    plt.legend()
    st.pyplot(plt)

except Exception as e:
    st.error(f"Prediction failed: {e}")
