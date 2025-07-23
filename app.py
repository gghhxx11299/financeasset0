import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom theme colors
COLORS = {
    'background': '#0E1117',
    'text': '#FAFAFA',
    'primary': '#00D1B2',
    'secondary': '#3273DC',
    'accent': '#FF3860',
    'positive': '#00C781',
    'negative': '#FF5050',
    'div_up': '#00D1B2',
    'div_down': '#FF3860'
}

# Apply custom theme
def set_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        .metric-container {{
            background-color: #1F2937;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #9CA3AF;
            font-weight: 500;
        }}
        .metric-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {COLORS['primary']};
            margin: 5px 0;
        }}
        .section-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {COLORS['primary']};
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid #374151;
        }}
        .info-card {{
            background-color: #1F2937;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid {COLORS['primary']};
        }}
        .prediction-card {{
            background-color: #1F2937;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #374151;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime

@dataclass
class KalmanFilterConfig:
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    P0: np.ndarray  # Initial error covariance
    x0: np.ndarray  # Initial state estimate
    G_min: float = -2.0  # Normalization bounds
    G_max: float = 2.0

@dataclass
class DivergenceStats:
    total_candles: int = 0
    failed_candles: int = 0
    failure_percentage: float = 0.0
    average_failure_magnitude: float = 0.0
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=10))

class KalmanFilter:
    def __init__(self, config: KalmanFilterConfig):
        self.Q = config.Q
        self.R = config.R
        self.P = config.P0
        self.x = config.x0
        self.A = np.eye(2)  # State transition matrix
        self.H = np.eye(2)  # Observation matrix
        self.G_min = config.G_min
        self.G_max = config.G_max
        self.residuals = []

    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update with measurement"""
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((2, 2))  # Fallback if matrix inversion fails
        
        residual = z - self.H @ self.x
        self.residuals.append(residual)
        
        self.x = self.x + K @ residual
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def normalize(self, G: float) -> float:
        """Normalize G value to 0-100 range"""
        return np.clip(100 * (G - self.G_min) / (self.G_max - self.G_min), 0, 100)

class DivergenceAnalyzer:
    def __init__(self, config: KalmanFilterConfig, timeframe: str):
        self.kf = KalmanFilter(config)
        self.timeframe = timeframe
        self.prev_div_up = None
        self.prev_div_down = None
        self.history = []
        self.stats = DivergenceStats()
        self.failure_threshold = 5.0  # 5% threshold for considering a failure

    def calculate_raw_G(self, current: Candle, previous: Optional[Candle]) -> Tuple[float, float]:
        """Calculate raw G_up and G_down values from candle structure"""
        if previous is None:
            return 0.0, 0.0

        # Current candle divergences
        if current.close > current.open:  # Bullish candle
            div_up = current.high - current.close    # Upper wick
            div_down = current.open - current.low    # Lower wick
        else:  # Bearish candle
            div_up = current.high - current.open     # Upper wick
            div_down = current.close - current.low   # Lower wick

        # Calculate deltas from previous candle
        delta_up = div_up - self.prev_div_up if self.prev_div_up is not None else 0
        delta_down = div_down - self.prev_div_down if self.prev_div_down is not None else 0

        # Store current divergences for next calculation
        self.prev_div_up = div_up
        self.prev_div_down = div_down

        # Calculate real move and G values
        real_move = abs(current.close - current.open)
        if real_move == 0:
            return 0.0, 0.0

        # G-values represent divergence strength relative to price movement
        G_up = delta_up / real_move
        G_down = delta_down / real_move

        return G_up, G_down

    def evaluate_divergence_failure(self, candle: Candle, Id_up: float, Id_down: float) -> Tuple[bool, float]:
        """
        Evaluate if the price action violated the divergence prediction.
        Returns:
            Tuple of (is_failure, failure_magnitude)
        """
        price_change = ((candle.close - candle.open) / candle.open) * 100
        
        # Determine expected direction based on divergence indices
        if Id_up > (100 - self.failure_threshold) and Id_down < self.failure_threshold:
            # Strong upward divergence expected
            expected_direction = 1
        elif Id_down > (100 - self.failure_threshold) and Id_up < self.failure_threshold:
            # Strong downward divergence expected
            expected_direction = -1
        else:
            # No strong divergence expected
            return False, 0.0
        
        # Check if price moved against expectation
        if (expected_direction == 1 and price_change < 0) or (expected_direction == -1 and price_change > 0):
            failure_magnitude = abs(price_change)
            return True, failure_magnitude
        
        return False, 0.0

    def process_candle(self, candle: Candle, prev_candle: Optional[Candle]) -> Dict:
        """Process a single candle through the divergence analysis pipeline"""
        # 1. Calculate raw divergence metrics from candle structure
        G_up_raw, G_down_raw = self.calculate_raw_G(candle, prev_candle)
        
        # 2. Apply Kalman Filter to smooth and track divergence metrics
        z = np.array([G_up_raw, G_down_raw])
        filtered_G = self.kf.update(z)
        predicted_G = self.kf.predict()
        
        # 3. Normalize to divergence indices (0-100 range)
        Id_up = self.kf.normalize(filtered_G[0])
        Id_down = self.kf.normalize(filtered_G[1])
        
        # 4. Evaluate prediction effectiveness
        is_failure, failure_mag = self.evaluate_divergence_failure(candle, Id_up, Id_down)
        
        # Update statistics
        self.stats.total_candles += 1
        if is_failure:
            self.stats.failed_candles += 1
            self.stats.recent_failures.append(failure_mag)
            if self.stats.total_candles > 0:
                self.stats.failure_percentage = (self.stats.failed_candles / self.stats.total_candles) * 100
                self.stats.average_failure_magnitude = np.mean(self.stats.recent_failures) if self.stats.recent_failures else 0.0
        
        result = {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "G_up_raw": G_up_raw,
            "G_down_raw": G_down_raw,
            "G_up_filtered": float(filtered_G[0]),
            "G_down_filtered": float(filtered_G[1]),
            "G_up_predicted": float(predicted_G[0]),
            "G_down_predicted": float(predicted_G[1]),
            "Id_up": float(Id_up),
            "Id_down": float(Id_down),
            "is_failure": is_failure,
            "failure_magnitude": float(failure_mag) if is_failure else 0.0,
            "direction_prediction": "Up" if Id_up > Id_down else "Down"
        }
        
        self.history.append(result)
        return result

    def get_stats_summary(self) -> Dict:
        """Return a summary of divergence statistics"""
        return {
            "total_candles": self.stats.total_candles,
            "failed_candles": self.stats.failed_candles,
            "failure_percentage": self.stats.failure_percentage,
            "average_failure_magnitude": self.stats.average_failure_magnitude,
            "recent_failures": list(self.stats.recent_failures)
        }

class MarketDataFetcher:
    @staticmethod
    def fetch_data(ticker: str, period: str, interval: str) -> List[Candle]:
        """Fetch OHLC data from yfinance and convert to Candle objects"""
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                raise ValueError(f"No data returned for {ticker} with period {period} and interval {interval}")
            
            candles = []
            
            for idx, row in data.iterrows():
                candles.append(Candle(
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    timestamp=idx.to_pydatetime()
                ))
            
            return candles
        except Exception as e:
            st.error(f"Failed to fetch data: {str(e)}")
            raise

class DivergenceModel:
    def __init__(self):
        self.ticker = "AAPL"
        self.period = "30d"
        self.timeframe = "15m"
        self.analyzers = {}
        self.current_results = None
        self.setup_analyzers()
    
    def setup_analyzers(self):
        """Initialize analyzers for all timeframes"""
        configs = self.get_configurations()
        self.analyzers = {
            tf: DivergenceAnalyzer(cfg, tf) 
            for tf, cfg in configs.items()
        }
    
    def get_configurations(self) -> Dict[str, KalmanFilterConfig]:
        """Return configurations for different timeframes"""
        base_Q = np.eye(2) * 0.01
        base_R = np.eye(2) * 0.1
        base_P0 = np.eye(2) * 0.1
        base_x0 = np.zeros(2)
        
        return {
            "1m": KalmanFilterConfig(
                Q=base_Q * 1.5,
                R=base_R * 0.8,
                P0=base_P0,
                x0=base_x0
            ),
            "5m": KalmanFilterConfig(
                Q=base_Q * 1.2,
                R=base_R,
                P0=base_P0,
                x0=base_x0
            ),
            "15m": KalmanFilterConfig(
                Q=base_Q,
                R=base_R,
                P0=base_P0,
                x0=base_x0
            ),
            "1h": KalmanFilterConfig(
                Q=base_Q * 0.8,
                R=base_R * 1.2,
                P0=base_P0,
                x0=base_x0
            ),
            "1d": KalmanFilterConfig(
                Q=base_Q * 0.5,
                R=base_R * 1.5,
                P0=base_P0,
                x0=base_x0
            ),
            "1wk": KalmanFilterConfig(
                Q=base_Q * 0.3,
                R=base_R * 2.0,
                P0=base_P0,
                x0=base_x0
            )
        }
    
    def process_timeframe(self, ticker: str, timeframe: str, period: str) -> pd.DataFrame:
        """Fetch and process data for a specific timeframe"""
        self.ticker = ticker
        self.timeframe = timeframe
        self.period = period
        
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "60m",
            "1d": "1d",
            "1wk": "1wk"
        }
        
        candles = MarketDataFetcher.fetch_data(
            ticker, 
            period=period, 
            interval=interval_map[timeframe]
        )
        
        results = []
        prev_candle = None
        
        analyzer = self.analyzers[timeframe]
        analyzer.history = []  # Reset history
        
        for candle in candles:
            result = analyzer.process_candle(candle, prev_candle)
            results.append(result)
            prev_candle = candle
        
        self.current_results = pd.DataFrame(results)
        return self.current_results
    
    def get_plot_data(self) -> Optional[Dict]:
        """Prepare data for plotting"""
        if self.current_results is None or self.current_results.empty:
            return None
            
        return {
            "timestamp": self.current_results['timestamp'].values,
            "close": self.current_results['close'].values,
            "G_up_filtered": self.current_results['G_up_filtered'].values,
            "G_down_filtered": self.current_results['G_down_filtered'].values,
            "Id_up": self.current_results['Id_up'].values,
            "Id_down": self.current_results['Id_down'].values,
            "is_failure": self.current_results['is_failure'].values,
            "failure_magnitude": self.current_results['failure_magnitude'].values,
            "direction_prediction": self.current_results['direction_prediction'].values,
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "period": self.period
        }
    
    def get_stats(self) -> Optional[Dict]:
        """Get divergence statistics for current timeframe"""
        if self.timeframe not in self.analyzers:
            return None
        return self.analyzers[self.timeframe].get_stats_summary()
    
    def get_last_prediction(self) -> Optional[Dict]:
        """Get the most recent prediction data"""
        if self.current_results is None or self.current_results.empty:
            return None
            
        last_row = self.current_results.iloc[-1].to_dict()
        return {
            "timestamp": last_row['timestamp'].strftime('%Y-%m-%d %H:%M'),
            "Id_up": last_row['Id_up'],
            "Id_down": last_row['Id_down'],
            "direction_prediction": last_row['direction_prediction'],
            "confidence": abs(last_row['Id_up'] - last_row['Id_down']),
            "G_up_filtered": last_row['G_up_filtered'],
            "G_down_filtered": last_row['G_down_filtered']
        }

def create_price_divergence_plot(plot_data: Dict) -> go.Figure:
    """Create interactive price and divergence plot using Plotly"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=(
                           f"{plot_data['ticker']} Price",
                           "Divergence Indices",
                           "Failure Magnitude"
                       ))
    
    # Price plot with failure markers
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['close'],
            name='Price',
            line=dict(color=COLORS['primary']),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Add failure markers if any
    failure_mask = plot_data['is_failure']
    if np.any(failure_mask):
        failure_times = np.array(plot_data['timestamp'])[failure_mask]
        failure_prices = np.array(plot_data['close'])[failure_mask]
        
        fig.add_trace(
            go.Scatter(
                x=failure_times,
                y=failure_prices,
                name='Divergence Failure',
                mode='markers',
                marker=dict(
                    color=COLORS['negative'],
                    size=8,
                    symbol='x'
                )
            ),
            row=1, col=1
        )
    
    # Divergence indices plot
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['Id_up'],
            name='Id_up (Bullish Divergence)',
            line=dict(color=COLORS['div_up']),
            mode='lines'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['Id_down'],
            name='Id_down (Bearish Divergence)',
            line=dict(color=COLORS['div_down']),
            mode='lines'
        ),
        row=2, col=1
    )
    
    # Add threshold lines
    fig.add_hline(
        y=95, line=dict(color='gray', dash='dash'),
        row=2, col=1
    )
    fig.add_hline(
        y=5, line=dict(color='gray', dash='dash'),
        row=2, col=1
    )
    
    # Failure magnitude plot
    if np.any(failure_mask):
        failure_magnitudes = np.array(plot_data['failure_magnitude'])
        failure_magnitudes[~failure_mask] = np.nan
        
        fig.add_trace(
            go.Bar(
                x=plot_data['timestamp'],
                y=failure_magnitudes,
                name='Failure Magnitude',
                marker_color=COLORS['negative']
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{plot_data['ticker']} {plot_data['timeframe']} - Last {plot_data['period']}",
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=50, r=50, b=50, t=100, pad=4),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Divergence Index (0-100)", row=2, col=1)
    fig.update_yaxes(title_text="Failure %", row=3, col=1)
    
    return fig

def create_metric(label: str, value: Union[str, float], delta: Optional[str] = None):
    """Create a styled metric component"""
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_prediction_card(prediction: Dict):
    """Create a styled card for the current prediction"""
    confidence = prediction['confidence']
    confidence_color = COLORS['positive'] if confidence > 20 else COLORS['accent'] if confidence > 10 else COLORS['negative']
    
    st.markdown(
        f"""
        <div class="prediction-card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">Current Market Prediction</h3>
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div>
                    <div style="color: #9CA3AF; font-size: 0.9rem;">Timestamp</div>
                    <div style="font-size: 1.1rem;">{prediction['timestamp']}</div>
                </div>
                <div>
                    <div style="color: #9CA3AF; font-size: 0.9rem;">Direction</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: {COLORS['div_up'] if prediction['direction_prediction'] == 'Up' else COLORS['div_down']}">
                        {prediction['direction_prediction']}
                    </div>
                </div>
                <div>
                    <div style="color: #9CA3AF; font-size: 0.9rem;">Confidence</div>
                    <div style="font-size: 1.1rem; color: {confidence_color}">{confidence:.1f}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div style="text-align: center;">
                    <div style="color: {COLORS['div_up']}">Bullish Divergence (Id_up)</div>
                    <div style="font-size: 1.2rem;">{prediction['Id_up']:.1f}</div>
                    <div style="color: #9CA3AF; font-size: 0.8rem;">G_up: {prediction['G_up_filtered']:.3f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {COLORS['div_down']}">Bearish Divergence (Id_down)</div>
                    <div style="font-size: 1.2rem;">{prediction['Id_down']:.1f}</div>
                    <div style="color: #9CA3AF; font-size: 0.8rem;">G_down: {prediction['G_down_filtered']:.3f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Kalman Filter Divergence Analyzer",
        page_icon="📊"
    )
    set_theme()
    
    st.title("📊 Structural Divergence Analysis Engine")
    st.markdown("""
    <div style="color: #9CA3AF; margin-bottom: 30px;">
        A sophisticated market structure analyzer that tracks price divergence patterns using Kalman filtering
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = DivergenceModel()
        st.session_state.data_loaded = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "1h", "1d", "1wk"],
                index=2
            )
        with col2:
            period = st.selectbox(
                "Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=2
            )
        
        if st.button("🚀 Analyze Market Structure", use_container_width=True):
            with st.spinner("Processing market data..."):
                try:
                    results = st.session_state.model.process_timeframe(ticker, timeframe, period)
                    st.session_state.results = results
                    st.session_state.plot_data = st.session_state.model.get_plot_data()
                    st.session_state.stats = st.session_state.model.get_stats()
                    st.session_state.prediction = st.session_state.model.get_last_prediction()
                    st.session_state.data_loaded = True
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    # Main content area
    if st.session_state.get('data_loaded', False):
        # Current prediction card
        st.markdown('<div class="section-header">📈 Current Market Structure</div>', unsafe_allow_html=True)
        create_prediction_card(st.session_state.prediction)
        
        # Performance metrics
        st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stats' in st.session_state and st.session_state.stats:
                stats = st.session_state.stats
                create_metric("Total Candles Analyzed", stats['total_candles'])
                create_metric("Failed Predictions", 
                            f"{stats['failed_candles']}",
                            f"{stats['failure_percentage']:.1f}% rate")
        
        with col2:
            if 'stats' in st.session_state and st.session_state.stats:
                stats = st.session_state.stats
                create_metric("Avg Failure Magnitude", 
                             f"{stats['average_failure_magnitude']:.2f}%")
                if stats['recent_failures']:
                    create_metric("Recent Failures", 
                                ", ".join([f"{x:.1f}%" for x in stats['recent_failures']]))
        
        # Visualization
        st.markdown('<div class="section-header">📉 Divergence Analysis</div>', unsafe_allow_html=True)
        if 'plot_data' in st.session_state and st.session_state.plot_data:
            fig = create_price_divergence_plot(st.session_state.plot_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Methodology explanation
        st.markdown('<div class="section-header">🔍 Methodology</div>', unsafe_allow_html=True)
        with st.expander("How this analysis works"):
            st.markdown("""
            <div class="info-card">
                <h3 style="color: {COLORS['primary']}; margin-top: 0;">Core Functionality</h3>
                <p>This engine analyzes market structure through a sophisticated divergence detection system:</p>
                
                <h4 style="color: {COLORS['primary']};">1. Data Processing</h4>
                <ul>
                    <li>Downloads high-quality historical price data</li>
                    <li>Analyzes each candle's wick/body structure for divergence patterns</li>
                    <li>Calculates both bullish (Id_up) and bearish (Id_down) divergence metrics</li>
                </ul>
                
                <h4 style="color: {COLORS['primary']};">2. Kalman Filter Application</h4>
                <ul>
                    <li>Smooths raw divergence data to filter out market noise</li>
                    <li>Adaptively tracks changes in market structure over time</li>
                    <li>Provides dynamic normalization of divergence metrics</li>
                </ul>
                
                <h4 style="color: {COLORS['primary']};">3. Predictive Analytics</h4>
                <ul>
                    <li>G-values measure divergence strength relative to price movement</li>
                    <li>Normalized indices (0-100 scale) show divergence extremity</li>
                    <li>Failure detection system identifies when price action contradicts structure</li>
                </ul>
            </div>
            
            <div class="info-card">
                <h3 style="color: {COLORS['primary']}; margin-top: 0;">Interpretation Guide</h3>
                
                <h4 style="color: {COLORS['primary']};">Directional Signals</h4>
                <ul>
                    <li><strong style="color: {COLORS['div_up']}">Id_up > Id_down</strong>: Suggests underlying bullish pressure</li>
                    <li><strong style="color: {COLORS['div_down']}">Id_down > Id_up</strong>: Suggests underlying bearish pressure</li>
                    <li>The greater the difference between indices, the stronger the signal</li>
                </ul>
                
                <h4 style="color: {COLORS['primary']};">Threshold Levels</h4>
                <ul>
                    <li><strong>Id_up > 95</strong>: Extreme bullish divergence</li>
                    <li><strong>Id_down > 95</strong>: Extreme bearish divergence</li>
                    <li>Values between 5-95 show moderate structural bias</li>
                </ul>
                
                <h4 style="color: {COLORS['primary']};">Practical Usage</h4>
                <p>This is not a trading system, but rather a <strong>structural analysis framework</strong>:</p>
                <ul>
                    <li>Use to confirm/deny other technical signals</li>
                    <li>Helps identify when market structure is strengthening/weakening</li>
                    <li>Failure patterns often precede trend reversals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Configure settings and click 'Analyze Market Structure' to begin")

if __name__ == "__main__":
    main()
