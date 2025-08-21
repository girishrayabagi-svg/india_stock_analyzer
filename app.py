import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# Helper Plot Functions
# -----------------------------
def plot_candles(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def load_data(ticker, years):
    """Download stock data from Yahoo Finance"""
    df = yf.download(ticker, period=f"{years}y", interval="1d")
    df.reset_index(inplace=True)
    return df
    ))
def plot_price_with_indicators(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name="Close Price"
    ))
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            mode='lines',
            name="MA20"
        ))
    if "MA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            mode='lines',
            name="MA50"
        ))
    fig.update_layout(
        title=f"{ticker} Price with Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rsi(df, ticker):
    if "RSI" not in df.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name="RSI"
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(
        title=f"{ticker} RSI",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_macd(df, ticker):
    if "MACD" not in df.columns or "Signal" not in df.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name="MACD"
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        mode='lines',
        name="Signal"
    ))
    fig.update_layout(
        title=f"{ticker} MACD",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_bollinger(df, ticker):
    if "BB_High" not in df.columns or "BB_Low" not in df.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name="Close"
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_High'],
        mode='lines',
        name="BB High",
        line=dict(color='red', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Low'],
        mode='lines',
        name="BB Low",
        line=dict(color='blue', dash='dot')
    ))
    fig.update_layout(
        title=f"{ticker} Bollinger Bands",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Indian Stock Analyzer & Predictor")

stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

ticker = st.selectbox("Choose a Stock", stock_list, index=0)
years = st.slider("Years of history", 1, 15, 5)

def add_indicators(df):
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Bollinger Bands
    from ta.volatility import BollingerBands
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()
    df['BB_Mid'] = indicator_bb.bollinger_mavg()

    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Load data
df_ind = load_data(ticker, years)

# Show Charts
plot_candles(df_ind, ticker)
plot_price_with_indicators(df_ind, ticker)
plot_rsi(df_ind, ticker)
plot_macd(df_ind, ticker)
plot_bollinger(df_ind, ticker)

# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("🔮 Stock Price Prediction")

data = df_ind.reset_index()
data["Date"] = pd.to_datetime(data["Date"])
data["Days"] = (data["Date"] - data["Date"].min()).dt.days

X = data[["Days"]]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Model RMSE:** {rmse:.2f}")

future_days = st.slider("Predict next N days", 1, 60, 15)
future = pd.DataFrame({"Days": np.arange(data["Days"].max() + 1, data["Days"].max() + future_days + 1)})
future_preds = model.predict(future)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data["Date"], y=y, mode='lines', name="Actual"))
fig2.add_trace(go.Scatter(x=X_test["Days"].apply(lambda x: data["Date"].min() + pd.Timedelta(days=x)),
                          y=y_pred, mode='lines', name="Predicted"))
future_dates = pd.date_range(start=data["Date"].max(), periods=future_days+1, freq='D')[1:]
fig2.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name="Future Forecast"))
fig2.update_layout(title=f"Prediction for {ticker}", template="plotly_dark", height=500)

st.plotly_chart(fig2, use_container_width=True)
