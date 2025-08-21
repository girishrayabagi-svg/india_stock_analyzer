# app.py
# Full-featured Indian Stock Analyzer — Indicators • Backtests • Predictions
# Save this file as app.py and run with Streamlit.

import warnings
warnings.filterwarnings("ignore")

import math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# technical analysis helpers
import ta

# scikit-learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# ----------------------------
# Config
# ----------------------------
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(page_title="Indian Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 Indian Stock Analyzer — Full Technical Suite, Backtests & Predictions")
st.caption("Educational — not financial advice. Data via Yahoo Finance (yfinance).")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: str, end: str, interval: str="1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()

def add_supertrend(df: pd.DataFrame, period: int=10, multiplier: float=3.0) -> pd.DataFrame:
    df = df.copy()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period).average_true_range()
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]
        prev_st = supertrend.iloc[i-1]
        prev_dir = direction.iloc[i-1]
        close = df["Close"].iloc[i]

        if prev_dir == 1:
            curr_st = min(curr_upper, prev_st) if close <= prev_st else curr_lower
            curr_dir = 1 if close <= prev_st else -1
        else:
            curr_st = max(curr_lower, prev_st) if close >= prev_st else curr_upper
            curr_dir = -1 if close >= prev_st else 1

        supertrend.iloc[i] = curr_st
        direction.iloc[i] = curr_dir

    df["SUPERTREND"] = supertrend
    df["SUPERTREND_DIR"] = direction  # 1=uptrend, -1=downtrend
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Moving averages
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_DIFF"] = macd.macd_diff()

    # RSI
    rsi = ta.momentum.RSIIndicator(df["Close"], window=14)
    df["RSI_14"] = rsi.rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    # Bollinger
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["Close"]

    # Volatility / Volume indicators
    df["ATR_14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["ADX_14"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    df["CCI_20"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()
    df["MFI_14"] = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

    # Ichimoku basics
    ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
    df["ICH_TENKAN"] = ichi.ichimoku_conversion_line()
    df["ICH_KIJUN"] = ichi.ichimoku_base_line()

    # Returns and forward target
    df["RET_1D"] = df["Close"].pct_change()
    df["FWD_RET_1D"] = df["Close"].pct_change().shift(-1)

    # Supertrend
    df = add_supertrend(df, period=10, multiplier=3.0)
    return df

# ----------------------------
# Backtesting helpers
# ----------------------------
def compute_strategy_signals(df: pd.DataFrame, name: str):
    sig = pd.Series(0, index=df.index, dtype=int)
    if name == "RSI (Buy<30, Sell>70)":
        sig[df["RSI_14"] < 30] = 1
        sig[df["RSI_14"] > 70] = -1
    elif name == "MACD (Signal Cross)":
        cross_up = (df["MACD"].shift(1) < df["MACD_SIGNAL"].shift(1)) & (df["MACD"] > df["MACD_SIGNAL"])
        cross_dn = (df["MACD"].shift(1) > df["MACD_SIGNAL"].shift(1)) & (df["MACD"] < df["MACD_SIGNAL"])
        sig[cross_up] = 1
        sig[cross_dn] = -1
    elif name == "Supertrend (Dir)":
        sig[df["SUPERTREND_DIR"] == 1] = 1
        sig[df["SUPERTREND_DIR"] == -1] = -1
    elif name == "SMA 20/50 Cross":
        cross_up = (df["SMA_20"].shift(1) < df["SMA_50"].shift(1)) & (df["SMA_20"] > df["SMA_50"])
        cross_dn = (df["SMA_20"].shift(1) > df["SMA_50"].shift(1)) & (df["SMA_20"] < df["SMA_50"])
        sig[cross_up] = 1
        sig[cross_dn] = -1
    return sig

def backtest_signals(df: pd.DataFrame, signals: pd.Series, trade_cost_bps: float = 5.0):
    cost = trade_cost_bps / 10000.0
    pos = signals.replace(0, method="ffill").fillna(0)
    ret = df["RET_1D"].fillna(0.0)
    strat_ret = pos.shift(1).fillna(0) * ret
    trades = pos.diff().abs().fillna(0)
    strat_ret -= trades * cost
    equity = (1 + strat_ret).cumprod()
    total_return = equity.iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days or 1
    cagr = (1 + total_return) ** (365.25 / days) - 1
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * math.sqrt(252)
    cum_max = equity.cummax()
    drawdown = (equity / cum_max - 1).min()
    wins = (strat_ret > 0).sum()
    losses = (strat_ret <= 0).sum()
    win_rate = wins / max(1, (wins + losses))
    stats = {
        "Total Return": f"{total_return*100:.2f}%",
        "CAGR": f"{cagr*100:.2f}%",
        "Sharpe (≈)": f"{sharpe:.2f}",
        "Max Drawdown": f"{drawdown*100:.2f}%",
        "Win Rate": f"{win_rate*100:.2f}%",
        "Trades (direction changes)": int(trades.sum())
    }
    return equity, strat_ret, stats

def show_backtest(df: pd.DataFrame, strategy_name: str, trade_cost_bps: float):
    signals = compute_strategy_signals(df, strategy_name)
    equity, strat_ret, stats = backtest_signals(df, signals, trade_cost_bps=trade_cost_bps)
    st.subheader("Backtest Performance")
    st.json(stats)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=(1+df["RET_1D"].fillna(0)).cumprod(), name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=df.index, y=equity, name=strategy_name))
    fig.update_layout(title="Equity Curve (₹1 initial)", xaxis_title="Date", yaxis_title="Equity (normalized)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Prediction / ML
# ----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df[[
        "SMA_20","SMA_50","EMA_12","EMA_26","MACD","MACD_SIGNAL","MACD_DIFF",
        "RSI_14","STOCH_K","STOCH_D","BB_WIDTH","ATR_14","ADX_14","CCI_20","MFI_14","OBV",
        "SUPERTREND","SUPERTREND_DIR"
    ]].copy()
    feats["RET_1D"] = df["RET_1D"]
    feats["RET_5D"] = df["Close"].pct_change(5)
    feats["RET_10D"] = df["Close"].pct_change(10)
    feats = feats.dropna()
    return feats

def train_predict(df: pd.DataFrame, horizon_days: int = 1):
    feats = build_features(df)
    if len(feats) < 60:
        raise ValueError("Not enough data for prediction. Increase history or use daily interval (1d).")
    y_class = (df["FWD_RET_1D"].loc[feats.index] > 0).astype(int)
    y_reg = df["FWD_RET_1D"].loc[feats.index]
    tscv = TimeSeriesSplit(n_splits=5)
    last_train_idx, last_test_idx = None, None
    for train_idx, test_idx in tscv.split(feats):
        last_train_idx, last_test_idx = train_idx, test_idx
    X_train, X_test = feats.iloc[last_train_idx], feats.iloc[last_test_idx]
    yc_train, yc_test = y_class.iloc[last_train_idx], y_class.iloc[last_test_idx]
    yr_train, yr_test = y_reg.iloc[last_train_idx], y_reg.iloc[last_test_idx]
    clf1 = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    clf2 = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=6)
    clf1.fit(X_train, yc_train)
    clf2.fit(X_train, yc_train)
    pred1 = clf1.predict(X_test)
    pred2 = clf2.predict(X_test)
    acc1 = accuracy_score(yc_test, pred1)
    acc2 = accuracy_score(yc_test, pred2)
    clf_best = clf2 if acc2 >= acc1 else clf1
    best_name = "RandomForestClassifier" if clf_best is clf2 else "LogisticRegression"
    rfr = RandomForestRegressor(n_estimators=400, random_state=42, max_depth=8)
    rfr.fit(X_train, yr_train)
    yreg_pred = rfr.predict(X_test)
    # Compute RMSE in a way compatible with all sklearn versions
    rmse = np.sqrt(mean_squared_error(yr_test, yreg_pred))
    X_latest = feats.iloc[[-1]]
    class_up_prob = clf_best.predict_proba(X_latest)[0][1] if hasattr(clf_best, "predict_proba") else float(clf_best.predict(X_latest)[0])
    next_day_ret = float(rfr.predict(X_latest)[0])
    results = {
        "Classifier Used": best_name,
        "Validation Accuracy (last fold)": f"{max(acc1, acc2)*100:.2f}%",
        "Regressor RMSE (last fold)": f"{rmse*100:.2f} % return",
        "Prob(Up Tomorrow)": f"{class_up_prob*100:.2f}%",
        "Predicted Next-Day Return": f"{next_day_ret*100:.2f}%"
    }
    importances = None
    if hasattr(clf_best, "feature_importances_"):
        importances = pd.Series(clf_best.feature_importances_, index=feats.columns).sort_values(ascending=False).head(15)
    return results, importances

# ----------------------------
# Streamlit UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    stock_list = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","WIPRO.NS","BHARTIARTL.NS"
    ]
    ticker = st.selectbox("Choose a Stock", stock_list, index=0)
    years = st.slider("Years of history", 1, 15, 5)
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    start = (datetime.now(IST) - timedelta(days=365*years)).date().isoformat()
    end = (datetime.now(IST) + timedelta(days=1)).date().isoformat()
    st.markdown("---")
    st.write("**Backtest**")
    strategy_name = st.selectbox("Strategy", ["RSI (Buy<30, Sell>70)","MACD (Signal Cross)","Supertrend (Dir)","SMA 20/50 Cross"])
    cost_bps = st.number_input("Per-Trade Cost (bps each side)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    st.markdown("---")
    if st.button("Load / Refresh Data"):
        st.session_state["reload"] = True

# Load data
try:
    df = load_price_data(ticker, start, end, interval=interval)
    if df.empty:
        st.error("No data returned. Check ticker/interval.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Add indicators
df_ind = add_indicators(df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview","Indicators & Charts","Backtest","Predict"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"₹{df_ind['Close'].iloc[-1]:.2f}")
    c2.metric("1D Change", f"{df_ind['RET_1D'].iloc[-1]*100:.2f}%")
    c3.metric("RSI-14", f"{df_ind['RSI_14'].iloc[-1]:.1f}")
    c4.metric("ADX-14", f"{df_ind['ADX_14'].iloc[-1]:.1f}")
    st.markdown("—")
    # candlestick
def plot_candles(df, ticker):
    import plotly.graph_objs as go
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

with tab2:
    st.subheader("Price with Key Indicators")
    plot_price_with_indicators(df_ind, ticker)
    st.markdown("### Oscillators")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.line_chart(df_ind[["RSI_14"]])
        st.line_chart(df_ind[["STOCH_K","STOCH_D"]])
    with c2:
        st.line_chart(df_ind[["MACD","MACD_SIGNAL"]])
        st.line_chart(df_ind[["BB_WIDTH"]])
    st.markdown("### Volatility & Volume")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.line_chart(df_ind[["ATR_14"]])
    with c4:
        st.line_chart(df_ind[["OBV"]])

with tab3:
    st.subheader(f"Strategy: {strategy_name}")
    show_backtest(df_ind, strategy_name, trade_cost_bps=cost_bps)

with tab4:
    st.subheader("Next-Day Direction & Return (Toy Model)")
    try:
        results, importances = train_predict(df_ind)
        st.json(results)
        if importances is not None:
            st.bar_chart(importances)
        st.caption("Models are simplistic; treat outputs as rough signals only.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Try increasing history length or switching interval to 1d.")

st.markdown("---")
st.caption("© 2025 — Built for local use, free. Uses open libraries: streamlit, yfinance, ta, scikit-learn, plotly.")
