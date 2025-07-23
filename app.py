import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import logging
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom theme colors
COLORS = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'primary': '#00FFAA',
    'secondary': '#0088FF',
    'accent': '#FF00AA',
    'positive': '#00FF88',
    'negative': '#FF4444',
    'div_up': '#00FFAA',
    'div_down': '#FF4444'
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
        .css-1d391kg {{
            background-color: {COLORS['background']};
        }}
        .st-bb {{
            background-color: {COLORS['background']};
        }}
        .st-at {{
            background-color: {COLORS['primary']};
        }}
        .st-ax {{
            color: {COLORS['text']};
        }}
        .metric-container {{
            background-color: #2A2A2A;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metric-label {{
            font-size: 1rem;
            color: {COLORS['text']};
            opacity: 0.8;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {COLORS['primary']};
        }}
        .positive-pnl {{
            color: {COLORS['positive']};
        }}
        .negative-pnl {{
            color: {COLORS['negative']};
        }}
        .trade-table {{
            background-color: #2A2A2A;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
        }}
        .disclaimer {{
            background-color: #2A2A2A;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            font-size: 0.8rem;
            color: #AAAAAA;
        }}
        .description {{
            background-color: #2A2A2A;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }}
        .author {{
            text-align: right;
            font-style: italic;
            color: {COLORS['secondary']};
            margin-top: 20px;
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
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    pnl: Optional[float]
    exit_reason: Optional[str]

@dataclass
class BacktestResult:
    total_trades: int
    profitable_trades: int
    win_rate: float
    avg_profit: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]

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
            st.warning("Kalman Filter Warning: Matrix inversion failed, using fallback")
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

class TradingAlgorithm:
    def __init__(self):
        self.current_position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        self.prev_id_up = None
        self.prev_id_down = None
    
    def process_candle(self, candle: Dict) -> Optional[Trade]:
        """Process a candle and return a Trade if one was closed"""
        if 'Id_up' not in candle or 'Id_down' not in candle:
            st.warning("Missing divergence indices in candle data")
            return None
            
        current_id_up = candle['Id_up']
        current_id_down = candle['Id_down']
        
        if self.prev_id_up is None or self.prev_id_down is None:
            self.prev_id_up = current_id_up
            self.prev_id_down = current_id_down
            return None
        
        # Calculate direction changes with threshold to avoid noise
        id_up_increased = current_id_up > (self.prev_id_up + 1.0)
        id_up_decreased = current_id_up < (self.prev_id_up - 1.0)
        id_down_increased = current_id_down > (self.prev_id_down + 1.0)
        id_down_decreased = current_id_down < (self.prev_id_down - 1.0)
        
        closed_trade = None
        
        # Check for entry signals
        if self.current_position is None:
            # Buy signal: Id_down decreases and Id_up increases
            if id_down_decreased and id_up_increased:
                self.enter_trade('long', candle)
            
            # Sell signal: Id_up decreases and Id_down increases
            elif id_up_decreased and id_down_increased:
                self.enter_trade('short', candle)
        
        # Check for exit signals - Modified to close opposite positions
        elif self.current_position == 'long':
            # Exit long when a short signal occurs
            if id_up_decreased and id_down_increased:
                closed_trade = self.exit_trade(candle, 'opposite_signal')
                # Enter short position immediately after closing long
                self.enter_trade('short', candle)
        
        elif self.current_position == 'short':
            # Exit short when a long signal occurs
            if id_down_decreased and id_up_increased:
                closed_trade = self.exit_trade(candle, 'opposite_signal')
                # Enter long position immediately after closing short
                self.enter_trade('long', candle)
        
        # Update previous values
        self.prev_id_up = current_id_up
        self.prev_id_down = current_id_down
        
        return closed_trade
    
    def enter_trade(self, direction: str, candle: Dict):
        """Enter a new trade"""
        self.current_position = direction
        self.entry_price = candle['close']
        self.entry_time = candle['timestamp']
    
    def exit_trade(self, candle: Dict, reason: str) -> Trade:
        """Exit the current trade and record it"""
        exit_price = candle['close']
        pnl = ((exit_price - self.entry_price) / self.entry_price) * 100
        if self.current_position == 'short':
            pnl *= -1
        
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=candle['timestamp'],
            entry_price=self.entry_price,
            exit_price=exit_price,
            direction=self.current_position,
            pnl=pnl,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        
        return trade
    
    def run_backtest(self, data: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical data"""
        self.reset()  # Clear any existing state
        
        for _, row in data.iterrows():
            candle = row.to_dict()
            self.process_candle(candle)
        
        # Close any open position at the end
        if self.current_position is not None:
            last_row = data.iloc[-1].to_dict()
            self.exit_trade(last_row, 'end_of_data')
        
        return self.get_results()
    
    def reset(self):
        """Reset the algorithm state"""
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        self.prev_id_up = None
        self.prev_id_down = None
    
    def get_results(self) -> BacktestResult:
        """Calculate backtest results"""
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                profitable_trades=0,
                win_rate=0,
                avg_profit=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=[]
            )
        
        pnls = [trade.pnl for trade in self.trades]
        profitable_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = profitable_trades / len(self.trades)
        avg_profit = np.mean(pnls)
        
        # Calculate max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
        
        return BacktestResult(
            total_trades=len(self.trades),
            profitable_trades=profitable_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades
        )

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
        """Calculate raw G_up and G_down values"""
        if previous is None:
            return 0.0, 0.0

        # Current candle divergences
        if current.close > current.open:  # Bullish
            div_up = current.high - current.close
            div_down = current.open - current.low
        else:  # Bearish
            div_up = current.high - current.open
            div_down = current.close - current.low

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
        """Process a single candle"""
        G_up_raw, G_down_raw = self.calculate_raw_G(candle, prev_candle)
        z = np.array([G_up_raw, G_down_raw])
        
        # Update Kalman filter
        filtered_G = self.kf.update(z)
        
        # Predict next values
        predicted_G = self.kf.predict()
        
        # Normalize to divergence index
        Id_up = self.kf.normalize(filtered_G[0])
        Id_down = self.kf.normalize(filtered_G[1])
        
        # Evaluate divergence effectiveness
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
            "failure_magnitude": float(failure_mag) if is_failure else 0.0
        }
        
        self.history.append(result)
        return result

    def get_stats_summary(self) -> Dict:
        """Return a summary of divergence statistics"""
        stats = {
            "total_candles": self.stats.total_candles,
            "failed_candles": self.stats.failed_candles,
            "failure_percentage": self.stats.failure_percentage,
            "average_failure_magnitude": self.stats.average_failure_magnitude,
            "recent_failures": list(self.stats.recent_failures)
        }
             
        return stats

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
            st.error(f"[ERROR] Failed to fetch data: {str(e)}")
            logger.error(f"Failed to fetch data: {str(e)}")
            raise Exception(f"Failed to fetch data: {str(e)}")

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
    
    def run_trading_backtest(self) -> BacktestResult:
        """Run the trading algorithm on current data"""
        if self.current_results is None or self.current_results.empty:
            raise ValueError("No data available for backtesting")
        
        # Verify required columns exist
        required_cols = ['timestamp', 'close', 'Id_up', 'Id_down']
        if not all(col in self.current_results.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.current_results.columns]
            raise ValueError(f"Missing required columns for backtest: {missing}")
        
        algorithm = TradingAlgorithm()
        result = algorithm.run_backtest(self.current_results)
        return result
    
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
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "period": self.period
        }
    
    def get_stats(self) -> Optional[Dict]:
        """Get divergence statistics for current timeframe"""
        if self.timeframe not in self.analyzers:
            return None
        return self.analyzers[self.timeframe].get_stats_summary()

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
            name='Id_up',
            line=dict(color=COLORS['div_up']),
            mode='lines'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['Id_down'],
            name='Id_down',
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
    fig.update_yaxes(title_text="Divergence Index", row=2, col=1)
    fig.update_yaxes(title_text="Failure %", row=3, col=1)
    
    return fig

def create_trade_analysis_plot(plot_data: Dict, backtest_result: BacktestResult) -> go.Figure:
    """Create interactive trade analysis plot using Plotly"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.7, 0.3],
                       subplot_titles=(
                           f"{plot_data['ticker']} Price with Trades",
                           "Divergence Indices"
                       ))
    
    # Price plot with trades
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
    
    # Add trade markers
    for trade in backtest_result.trades:
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                name='Entry' if trade == backtest_result.trades[0] else None,
                mode='markers',
                marker=dict(
                    color=COLORS['positive'] if trade.direction == 'long' else COLORS['negative'],
                    size=10,
                    symbol='triangle-up'
                ),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"""
                Entry {trade.direction}<br>
                Time: {trade.entry_time.strftime('%Y-%m-%d %H:%M')}<br>
                Price: {trade.entry_price:.2f}
                """
            ),
            row=1, col=1
        )
        
        # Exit marker if exists
        if trade.exit_time:
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    name='Exit' if trade == backtest_result.trades[0] else None,
                    mode='markers',
                    marker=dict(
                        color=COLORS['positive'] if trade.pnl > 0 else COLORS['negative'],
                        size=10,
                        symbol='triangle-down'
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"""
                    Exit {trade.direction}<br>
                    Time: {trade.exit_time.strftime('%Y-%m-%d %H:%M')}<br>
                    Price: {trade.exit_price:.2f}<br>
                    PnL: {trade.pnl:.2f}%
                    """
                ),
                row=1, col=1
            )
    
    # Divergence indices plot
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['Id_up'],
            name='Id_up',
            line=dict(color=COLORS['div_up']),
            mode='lines'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data['timestamp'],
            y=plot_data['Id_down'],
            name='Id_down',
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
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text=f"Trade Analysis - {backtest_result.total_trades} Trades (Win Rate: {backtest_result.win_rate:.1%})",
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
    fig.update_yaxes(title_text="Divergence Index", row=2, col=1)
    
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

def create_trade_table(trades: List[Trade]):
    """Create a styled table of trades"""
    if not trades:
        st.warning("No trades to display")
        return
    
    # Prepare trade data
    trade_data = []
    for i, trade in enumerate(trades):
        pnl_class = "positive-pnl" if trade.pnl > 0 else "negative-pnl"
        trade_data.append({
            "#": i+1,
            "Direction": trade.direction,
            "Entry Time": trade.entry_time.strftime('%Y-%m-%d %H:%M'),
            "Entry Price": f"{trade.entry_price:.2f}",
            "Exit Time": trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else "-",
            "Exit Price": f"{trade.exit_price:.2f}" if trade.exit_price else "-",
            "PnL (%)": f"<span class='{pnl_class}'>{trade.pnl:.2f}%</span>",
            "Reason": trade.exit_reason
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(trade_data)
    
    # Display styled table
    st.markdown(
        f"""
        <div class="trade-table">
            {df.to_html(escape=False, index=False)}
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
    
    st.title("📊 Kalman Filter Divergence Analyzer")
    st.markdown("""
    <style>
    .title {
        color: #00FFAA;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Description section
    with st.expander("📝 Description", expanded=True):
        st.markdown("""
        <div class="description">
        <h3>About This Tool</h3>
        <p>Kalman Filter Divergence Analysis (KFDA) is a novel stock forecasting framework that blends price action structure with signal processing to model and predict directional market behavior. Unlike traditional indicators that rely heavily on price, volume, or trend-following logic, KFDA focuses on structural divergence—capturing the misalignment between actual price behavior and expected movement based on candle formations.</p>
        
        <h4>Core Innovation: The Id Metric:</h4>
        <ul>
            <li>Id_up represents the probability (expressed as a percentage) that the stock is expected to rise based on the filtered divergence signals.</li>
            <li>Id_down mirrors this by showing the probability of a downward move.</li>
        </ul>
        <h4> Description </h4>
        <ul>
            <li>Backtesting capability with trade simulation</li>
            <li>Visualization of divergence failures and performance metrics</li>
            <li>Interactive charts with Plotly</li>
        </ul>
        <h4> Performance Dashboard
         The KFDA tool provides a complete dashboard that includes
        </h4>
        <ul>
            <li> Candle analysis metrics (total candles, divergence failures</li>
            <li>Backtested performance (Sharpe ratio, max drawdown, profit stats)</li>
            <li>Real-time probabilistic signals</li>
        </ul>
        <h4>How It Works:</h4>
        <p>The algorithm calculates two divergence indices (Id_up and Id_down) that measure the 
        relationship between price movements and their divergences. When these indices cross certain 
        thresholds, they generate trading signals. The Kalman filter helps smooth these signals 
        and reduce noise.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = DivergenceModel()
        st.session_state.data_loaded = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
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
        
        if st.button("🚀 Process Data", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    results = st.session_state.model.process_timeframe(ticker, timeframe, period)
                    plot_data = st.session_state.model.get_plot_data()
                    stats = st.session_state.model.get_stats()
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.plot_data = plot_data
                    st.session_state.stats = stats
                    
                    # Enable backtest button
                    st.session_state.data_loaded = True
                    st.success("Data processed successfully!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
        
        if st.button("📈 Run Backtest", 
                    use_container_width=True,
                    disabled=not st.session_state.get('data_loaded', False)):
            with st.spinner("Running backtest..."):
                try:
                    backtest_result = st.session_state.model.run_trading_backtest()
                    st.session_state.backtest_result = backtest_result
                    st.success("Backtest completed!")
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
        
        # Disclaimer section
        st.markdown("""
        <div class="disclaimer">
        <h4>⚠️ Disclaimer</h4>
        <p>This tool is for educational and research purposes only. The analysis provided 
        should not be considered as financial advice. Trading financial markets involves 
        risk, and past performance is not indicative of future results. Always conduct 
        your own research before making any trading decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Author credit
        st.markdown("""
        <div class="author">
        <p>Developed by Gebreal Mulugeta</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.get('data_loaded', False):
        # Stats and metrics
        st.subheader("📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'stats' in st.session_state and st.session_state.stats:
                stats = st.session_state.stats
                create_metric("Total Candles", stats['total_candles'])
                create_metric("Failed Predictions", 
                            f"{stats['failed_candles']}",
                            f"{stats['failure_percentage']:.1f}%")
        
        with col2:
            if 'stats' in st.session_state and st.session_state.stats:
                stats = st.session_state.stats
                create_metric("Avg Failure Magnitude", 
                             f"{stats['average_failure_magnitude']:.2f}%")
                if stats['recent_failures']:
                    create_metric("Recent Failures", 
                                ", ".join([f"{x:.1f}%" for x in stats['recent_failures']]))
        
        with col3:
            if 'backtest_result' in st.session_state:
                result = st.session_state.backtest_result
                create_metric("Total Trades", result.total_trades)
                create_metric("Profitable Trades", 
                            result.profitable_trades,
                            f"{result.win_rate:.1%}")
        
        with col4:
            if 'backtest_result' in st.session_state:
                result = st.session_state.backtest_result
                create_metric("Avg Profit", f"{result.avg_profit:.2f}%")
                create_metric("Max Drawdown", f"{result.max_drawdown:.2f}%")
                create_metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        
        # Plot area
        st.subheader("📈 Analysis Charts")
        if 'plot_data' in st.session_state and st.session_state.plot_data:
            plot_data = st.session_state.plot_data
            
            # Create tabs for different plot views
            tab1, tab2 = st.tabs(["Price and Divergence", "Trade Analysis"])
            
            with tab1:
                fig = create_price_divergence_plot(plot_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if 'backtest_result' in st.session_state:
                    fig = create_trade_analysis_plot(plot_data, st.session_state.backtest_result)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Run backtest to see trade analysis")
        
        # Results table
        st.subheader("📋 Recent Results")
        if 'results' in st.session_state and st.session_state.results is not None:
            # Show last 200 results
            display_df = st.session_state.results.tail(200).copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['close'] = display_df['close'].round(2)
            display_df['Id_up'] = display_df['Id_up'].round(1)
            display_df['Id_down'] = display_df['Id_down'].round(1)
            display_df['failure_magnitude'] = display_df['failure_magnitude'].round(2)
            
            st.dataframe(
                display_df[['timestamp', 'close', 'Id_up', 'Id_down', 'is_failure', 'failure_magnitude']],
                height=400,
                use_container_width=True
            )
        
        # Trade details
        if 'backtest_result' in st.session_state:
            st.subheader("💼 Trade Details")
            create_trade_table(st.session_state.backtest_result.trades)
    else:
        st.info("👈 Enter settings and click 'Process Data' to begin analysis")

if __name__ == "__main__":
    main()
