import numpy as np
import pandas as pd
import yfinance as yf
from scipy import interpolate, stats
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import streamlit as st
import warnings
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import KroghInterpolator
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# SELF-CONTAINED HDFM MODEL
# ======================
class RobustHDFM:
    def __init__(self, base_window=50, epsilon=1e-8):
        self.base_window = base_window
        self.epsilon = epsilon
        self.error_history = {'up': [], 'down': []}
        self.current_regime = 0
        self.circuit_breaker = False

    def _dynamic_window(self, volatility_ratio):
        return max(20, min(100, int(self.base_window / np.sqrt(volatility_ratio))))

    def _robust_zscore(self, series, window):
        rolling = series.rolling(window)
        q1 = rolling.quantile(0.25)
        q3 = rolling.quantile(0.75)
        iqr = q3 - q1
        return (series - rolling.median()) / (iqr + self.epsilon)

    def _calculate_atr(self, high, low, close, length=14):
        try:
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            hl = high - low
            hc = np.abs(high[1:] - close[:-1])
            lc = np.abs(low[1:] - close[:-1])
            hc = np.concatenate([[np.nan], hc])
            lc = np.concatenate([[np.nan], lc])
            tr = np.maximum.reduce([hl, hc, lc])
            atr = pd.Series(tr).rolling(length, min_periods=1).mean()
            return atr
        except Exception as e:
            logger.error(f"ATR calculation error: {str(e)}")
            return pd.Series(np.nan, index=range(len(high)))

    def _detect_anomalies(self, df):
        if df.empty or len(df) < 100: return False
        recent = df.iloc[-1]
        avg = df.rolling(100).mean().iloc[-1]
        atr = self._calculate_atr(df['High'].values, df['Low'].values, df['Close'].values)
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        volume_spike = recent['Volume'] > 5 * avg['Volume']
        price_gap = abs(recent['Close'] - recent['Open']) > 3 * current_atr
        volatility_spike = current_atr > 2 * avg_atr
        return volume_spike or price_gap or volatility_spike

    def _detect_regime(self, prices):
        if len(prices) < 100: return 0
        returns = np.diff(np.log(prices))
        std = np.std(returns)
        if std > 0.02: return 2
        elif std > 0.01: return 1
        return 0

    def calculate_divergences(self, ohlc_df):
        if ohlc_df.empty or len(ohlc_df) < 20:
            return pd.DataFrame()
        try:
            df = ohlc_df.copy()
            df['ATR'] = self._calculate_atr(df['High'].values, df['Low'].values, df['Close'].values)
            if len(df['ATR']) != len(df):
                df['ATR'] = df['ATR'].reindex(df.index, method='ffill')
            volatility_ratio = df['ATR'].iloc[-1] / (df['ATR'].mean() + self.epsilon)
            window = self._dynamic_window(volatility_ratio)
            if self._detect_anomalies(df):
                self.circuit_breaker = True
                st.warning("Market anomaly detected - entering safe mode")
                return pd.DataFrame()
            df['D_up'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['D_down'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['R'] = abs(df['Close'] - df['Open'])
            df['delta_D_up'] = df['D_up'].diff().clip(-3*df['ATR'], 3*df['ATR'])
            df['delta_D_down'] = df['D_down'].diff().clip(-3*df['ATR'], 3*df['ATR'])
            df['G_up'] = df['delta_D_up'] / (df['R'] + self.epsilon)
            df['G_down'] = df['delta_D_down'] / (df['R'] + self.epsilon)
            df['Id_up'] = self._robust_zscore(df['G_up'], window)
            df['Id_down'] = self._robust_zscore(df['G_down'], window)
            self.current_regime = self._detect_regime(df['Close'].values)
            df['C'] = abs(df['Id_up'] - df['Id_down']) * (1 - 0.2*self.current_regime)
            return df
        except Exception as e:
            logger.error(f"Divergence calculation error: {str(e)}")
            return pd.DataFrame()

    # ======================
    # Hierarchical Newton Interpolation for Id values
    # ======================
    def hierarchical_newton_interpolation(self, tf_keys, multi_tf_data, value_key):
        x = []
        y = []
        for i, tf in enumerate(tf_keys):
            tf_df = multi_tf_data[tf]
            if not tf_df.empty:
                x.append(i)
                y.append(tf_df[value_key].iloc[-1])
        if len(x) < 2:
            return np.mean(y) if y else 0
        interpolator = KroghInterpolator(x, y)
        next_x = max(x) + 1
        return float(interpolator(next_x))

    # ======================
    # Directional Confidence Calculation (softmax logic)
    # ======================
    def calc_directional_confidence(self, Id_up, Id_down):
        exp_up = np.exp(Id_up)
        exp_down = np.exp(Id_down)
        total = exp_up + exp_down
        I_up = exp_up / total if total != 0 else 0.5
        I_down = exp_down / total if total != 0 else 0.5
        direction = 'Up' if I_up > I_down else 'Down'
        confidence = abs(Id_up - Id_down) / (abs(Id_up) + abs(Id_down) + self.epsilon)
        return direction, confidence, I_up, I_down

    # ======================
    # Noise Amplification Index
    # ======================
    def calc_noise_index(self, multi_tf_data, value_key):
        id_vals = [df[value_key].iloc[-1] for df in multi_tf_data.values() if not df.empty]
        if len(id_vals) < 2:
            return 0
        return float(np.std(id_vals))

    # ======================
    # Final Output Construction
    # ======================
    def generate_hdfm_output(self, tf_keys, multi_tf_data):
        Id_up_interp = self.hierarchical_newton_interpolation(tf_keys, multi_tf_data, 'Id_up')
        Id_down_interp = self.hierarchical_newton_interpolation(tf_keys, multi_tf_data, 'Id_down')
        direction, confidence, I_up, I_down = self.calc_directional_confidence(Id_up_interp, Id_down_interp)
        noise_index = self.calc_noise_index(multi_tf_data, 'Id_up')
        primary_tf = tf_keys[0]
        primary_df = multi_tf_data[primary_tf]
        G_up = float(primary_df['G_up'].iloc[-1]) if not primary_df.empty else 0
        G_down = float(primary_df['G_down'].iloc[-1]) if not primary_df.empty else 0
        Id_up = float(primary_df['Id_up'].iloc[-1]) if not primary_df.empty else 0
        Id_down = float(primary_df['Id_down'].iloc[-1]) if not primary_df.empty else 0
        return {
            "Direction": direction,
            "Confidence": round(confidence, 3),
            "G_up": round(G_up, 4),
            "G_down": round(G_down, 4),
            "Id_up": round(Id_up, 4),
            "Id_down": round(Id_down, 4),
            "I_up": round(I_up, 3),
            "I_down": round(I_down, 3),
            "Noise Index": round(noise_index, 3)
        }

    def forecast(self, current_price, metrics):
        try:
            regime_penalty = 0.1 * self.current_regime
            B_up = current_price * (1 + metrics['G_up'] * metrics['Id_up'] * (1 - regime_penalty))
            B_down = current_price * (1 + metrics['G_down'] * metrics['Id_down'] * (1 - regime_penalty))
            confidence = metrics.get('C', 0.5)
            position_size = min(0.5, max(0.05, confidence * 0.5))
            return B_up, B_down, position_size
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return current_price, current_price, 0

    def adaptive_error_correction(self, current_price, predictions, actual_future):
        try:
            B_up, B_down, _ = predictions
            r_up = np.clip((actual_future - B_up)/actual_future, -0.2, 0.2)
            r_down = np.clip((actual_future - B_down)/actual_future, -0.2, 0.2)
            self.error_history['up'].append(r_up * 0.95**len(self.error_history['up']))
            self.error_history['down'].append(r_down * 0.95**len(self.error_history['down']))
            for k in self.error_history:
                if len(self.error_history[k]) > self.base_window:
                    self.error_history[k] = self.error_history[k][-self.base_window:]
            Q_up = np.median(self.error_history['up'])
            Q_down = np.median(self.error_history['down'])
            T_up = stats.iqr(self.error_history['up'])
            T_down = stats.iqr(self.error_history['down'])
            adj_factor = min(1, len(self.error_history['up'])/self.base_window)
            B_up_adj = current_price * (1 - Q_up * adj_factor)
            B_down_adj = current_price * (1 - Q_down * adj_factor)
            return (B_up_adj - T_up, B_up_adj + T_up), (B_down_adj - T_down, B_down_adj + T_down)
        except Exception as e:
            logger.error(f"Error correction error: {str(e)}")
            return (current_price, current_price), (current_price, current_price)

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    st.set_page_config(layout="wide")
    st.title("🔮 Hierarchical Divergence Forecast Model (HDFM)")

    with st.sidebar:
        st.header("Configuration")
        symbol = st.selectbox("Asset", ["AAPL", "TSLA", "BTC-USD", "ETH-USD", "SPY"])
        timeframes = st.multiselect("Timeframes", ["5m","15m","30m","1h","4h","1d"], default=["15m","1h"])
        risk_level = st.slider("Risk Level", 1, 5, 3)

        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                try:
                    model = RobustHDFM(base_window=30 + 20*(5-risk_level))
                    data = {}
                    for tf in timeframes:
                        data[tf] = yf.download(symbol, period="60d", interval=tf)
                        if not data[tf].empty:
                            data[tf] = model.calculate_divergences(data[tf])

                    if model.circuit_breaker:
                        st.error("Trading suspended due to market anomaly")
                        return

                    # Generate and show HDFM output table
                    results = model.generate_hdfm_output(timeframes, data)
                    st.subheader("📊 HDFM Output")
                    st.dataframe(pd.DataFrame([results]))

                    # Forecast price targets
                    primary_tf = timeframes[0]
                    primary_data = data.get(primary_tf, pd.DataFrame())
                    if primary_data.empty:
                        st.error("No valid data available for primary timeframe")
                        return
                    last = primary_data.iloc[-1]

                    G_up, G_down = results["G_up"], results["G_down"]
                    Id_up, Id_down = results["Id_up"], results["Id_down"]
                    C = results["Confidence"]
                    preds = model.forecast(
                        current_price=last['Close'],
                        metrics={'G_up': G_up, 'G_down': G_down, 'Id_up': Id_up, 'Id_down': Id_down, 'C': C}
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${last['Close']:.2f}")
                        st.metric("Direction", results["Direction"])
                        st.metric("Noise Index", f"{results['Noise Index']:.3f}")
                    with col2:
                        st.metric("Up Target", f"${preds[0]:.2f}", delta=f"{(preds[0]/last['Close']-1)*100:.1f}%")
                        st.metric("Down Target", f"${preds[1]:.2f}", delta=f"{(preds[1]/last['Close']-1)*100:.1f}%")
                    with col3:
                        st.metric("Confidence", f"{results['Confidence']*100:.1f}%")
                        st.metric("Position Size", f"{preds[2]*100:.1f}% of portfolio")
                        if model.current_regime == 2:
                            st.warning("High volatility - reduce position size")

                    # Visualizations
                    st.subheader("Price and Signals")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(primary_data.index, primary_data['Close'], label='Price', color='black')
                    up_signals = primary_data[primary_data['Id_up'] > primary_data['Id_down']]
                    if not up_signals.empty:
                        ax.scatter(up_signals.index, up_signals['Close'], label='Up Signal', color='green', marker='^')
                    down_signals = primary_data[primary_data['Id_up'] < primary_data['Id_down']]
                    if not down_signals.empty:
                        ax.scatter(down_signals.index, down_signals['Close'], label='Down Signal', color='red', marker='v')
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logger.exception("Analysis error")

if __name__ == "__main__":
    plt.style.use('ggplot')
    main()
