import numpy as np
import pandas as pd
import yfinance as yf
from scipy import interpolate, stats
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

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
        """Volatility-adaptive window sizing"""
        return max(20, min(100, int(self.base_window / np.sqrt(volatility_ratio))))
    
    def _robust_zscore(self, series, window):
        """IQR-based normalization resistant to outliers"""
        rolling = series.rolling(window)
        q1 = rolling.quantile(0.25)
        q3 = rolling.quantile(0.75)
        iqr = q3 - q1
        return (series - rolling.median()) / (iqr + self.epsilon)
    
    def _calculate_atr(self, high, low, close, length=14):
        """Manual ATR calculation without pandas-ta"""
        # Ensure we have pandas Series with proper index
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        # Calculate True Range components
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        # Combine components and calculate ATR
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(length).mean()
    
    def _detect_anomalies(self, df):
        """Market shock detection using manual calculations"""
        if df.empty or len(df) < 100:
            return False
            
        recent = df.iloc[-1]
        avg = df.rolling(100).mean().iloc[-1]
        
        atr = self._calculate_atr(df['High'], df['Low'], df['Close'])
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        
        volume_spike = recent['Volume'] > 5 * avg['Volume']
        price_gap = abs(recent['Close'] - recent['Open']) > 3 * current_atr
        volatility_spike = current_atr > 2 * avg_atr
        
        return volume_spike or price_gap or volatility_spike
    
    def _detect_regime(self, prices):
        """Simplified regime detection without HMM"""
        if len(prices) < 100:
            return 0
        
        returns = np.diff(np.log(prices))
        std = np.std(returns)
        
        if std > 0.02:  # 2% daily volatility threshold
            return 2  # High volatility
        elif std > 0.01:
            return 1  # Medium volatility
        return 0  # Low volatility
    
    def calculate_divergences(self, ohlc_df):
        if ohlc_df.empty or len(ohlc_df) < 20:
            return pd.DataFrame()
            
        df = ohlc_df.copy()
        
        # Volatility metrics with manual ATR
        df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
        volatility_ratio = df['ATR'].iloc[-1] / df['ATR'].mean()
        window = self._dynamic_window(volatility_ratio)
        
        # Circuit breaker check
        if self._detect_anomalies(df):
            self.circuit_breaker = True
            st.warning("Market anomaly detected - entering safe mode")
            return pd.DataFrame()
        
        # Divergence calculations
        df['D_up'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['D_down'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['R'] = abs(df['Close'] - df['Open'])
        
        # Robust G-values with clipping
        df['delta_D_up'] = df['D_up'].diff().clip(-3*df['ATR'], 3*df['ATR'])
        df['delta_D_down'] = df['D_down'].diff().clip(-3*df['ATR'], 3*df['ATR'])
        df['G_up'] = df['delta_D_up'] / (df['R'] + self.epsilon)
        df['G_down'] = df['delta_D_down'] / (df['R'] + self.epsilon)
        
        # Non-parametric normalization
        df['Id_up'] = self._robust_zscore(df['G_up'], window)
        df['Id_down'] = self._robust_zscore(df['G_down'], window)
        
        # Regime detection
        self.current_regime = self._detect_regime(df['Close'].values)
        
        # Confidence with regime penalty
        df['C'] = abs(df['Id_up'] - df['Id_down']) * (1 - 0.2*self.current_regime)
        
        return df
    
    def hierarchical_interpolation(self, multi_tf_data):
        # Manual feature combination without PCA
        tf_keys = sorted(multi_tf_data.keys())
        weighted_G_up = 0
        weighted_G_down = 0
        total_weight = 0
        
        # Simple weighted average based on timeframe "importance"
        for i, tf in enumerate(tf_keys):
            tf_df = multi_tf_data[tf]
            if tf_df.empty:
                continue
            weight = 1 / (i + 1)  # Higher weight for shorter timeframes
            weighted_G_up += tf_df['G_up'].mean() * weight
            weighted_G_down += tf_df['G_down'].mean() * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0, 0
            
        return weighted_G_up/total_weight, weighted_G_down/total_weight
    
    def forecast(self, current_price, metrics):
        # Regime-adjusted predictions
        regime_penalty = 0.1 * self.current_regime
        B_up = current_price * (1 + metrics['G_up'] * metrics['Id_up'] * (1 - regime_penalty))
        B_down = current_price * (1 + metrics['G_down'] * metrics['Id_down'] * (1 - regime_penalty))
        
        # Simplified position sizing
        confidence = metrics.get('C', 0.5)
        position_size = min(0.5, max(0.05, confidence * 0.5))
        return B_up, B_down, position_size
    
    def adaptive_error_correction(self, current_price, predictions, actual_future):
        B_up, B_down, _ = predictions
        
        # Safe error ratios
        r_up = np.clip((actual_future - B_up)/actual_future, -0.2, 0.2)
        r_down = np.clip((actual_future - B_down)/actual_future, -0.2, 0.2)
        
        # Store with decay factor
        self.error_history['up'].append(r_up * 0.95**len(self.error_history['up']))
        self.error_history['down'].append(r_down * 0.95**len(self.error_history['down']))
        
        # Trim and calculate
        for k in self.error_history:
            if len(self.error_history[k]) > self.base_window:
                self.error_history[k] = self.error_history[k][-self.base_window:]
        
        Q_up = np.median(self.error_history['up'])
        Q_down = np.median(self.error_history['down'])
        T_up = stats.iqr(self.error_history['up'])
        T_down = stats.iqr(self.error_history['down'])
        
        # Confidence-based adjustment
        adj_factor = min(1, len(self.error_history['up'])/self.base_window)
        B_up_adj = current_price * (1 - Q_up * adj_factor)
        B_down_adj = current_price * (1 - Q_down * adj_factor)
        
        return (B_up_adj - T_up, B_up_adj + T_up), (B_down_adj - T_down, B_down_adj + T_down)

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    st.set_page_config(layout="wide")
    st.title("🔒 Robust HDFM Trading System (Self-Contained Version)")
    
    with st.sidebar:
        st.header("Configuration")
        symbol = st.selectbox("Asset", ["AAPL", "TSLA", "BTC-USD", "ETH-USD", "SPY"])
        timeframes = st.multiselect("Timeframes", ["5m","15m","30m","1h","4h","1d"], default=["15m","1h"])
        risk_level = st.slider("Risk Level", 1, 5, 3)
        
        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                # Initialize with risk-adaptive window
                model = RobustHDFM(base_window=30 + 20*(5-risk_level))
                
                # Download and process data
                data = {}
                for tf in timeframes:
                    data[tf] = yf.download(symbol, period="60d", interval=tf)
                    if not data[tf].empty:
                        data[tf] = model.calculate_divergences(data[tf])
                
                if model.circuit_breaker:
                    st.error("Trading suspended due to market anomaly")
                else:
                    # Generate predictions
                    primary_tf = timeframes[0]
                    primary_data = data[primary_tf]
                    
                    if primary_data.empty:
                        st.error("No valid data available for analysis")
                        return
                    
                    last = primary_data.iloc[-1]
                    
                    # Get interpolated signals
                    G_up, G_down = model.hierarchical_interpolation(data)
                    
                    # Make forecast
                    preds = model.forecast(
                        current_price=last['Close'],
                        metrics={
                            'G_up': G_up,
                            'G_down': G_down,
                            'Id_up': last['Id_up'],
                            'Id_down': last['Id_down'],
                            'C': last['C']
                        }
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${last['Close']:.2f}")
                        st.metric("Market Regime", 
                                f"{['Low','Medium','High'][model.current_regime]} Volatility",
                                delta=f"{['🟢','🟡','🔴'][model.current_regime]}")
                    
                    with col2:
                        st.metric("Adjusted Up Target", 
                                f"${preds[0]:.2f}",
                                delta=f"{(preds[0]/last['Close']-1)*100:.1f}%")
                        st.metric("Adjusted Down Target", 
                                f"${preds[1]:.2f}",
                                delta=f"{(preds[1]/last['Close']-1)*100:.1f}%")
                    
                    with col3:
                        st.metric("Recommended Position", 
                                f"{preds[2]*100:.1f}% of portfolio",
                                help="Confidence-based sizing")
                        
                        if model.current_regime == 2:
                            st.warning("High volatility - consider reducing position size")
                    
                    # Visualizations
                    st.subheader("Price and Signals")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(primary_data.index, primary_data['Close'], label='Price', color='black')
                    ax.scatter(primary_data.index, 
                              np.where(primary_data['Id_up'] > primary_data['Id_down'], 
                                      primary_data['Close'], np.nan),
                              label='Up Signal', color='green', marker='^')
                    ax.scatter(primary_data.index, 
                              np.where(primary_data['Id_up'] < primary_data['Id_down'], 
                                      primary_data['Close'], np.nan),
                              label='Down Signal', color='red', marker='v')
                    ax.legend()
                    st.pyplot(fig)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
