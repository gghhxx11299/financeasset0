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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Verify we have required data
        if 'Id_up' not in candle or 'Id_down' not in candle:
            st.warning("Missing divergence indices in candle data")
            return None
            
        current_id_up = candle['Id_up']
        current_id_down = candle['Id_down']
        
        # Initialize previous values if this is the first candle
        if self.prev_id_up is None or self.prev_id_down is None:
            self.prev_id_up = current_id_up
            self.prev_id_down = current_id_down
            return None
        
        # Debug print current values
        st.write(f"\nProcessing candle at {candle['timestamp']}")
        st.write(f"Current Id_up: {current_id_up:.2f}, Previous: {self.prev_id_up:.2f}")
        st.write(f"Current Id_down: {current_id_down:.2f}, Previous: {self.prev_id_down:.2f}")
        
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
                st.write("BUY SIGNAL detected")
                self.enter_trade('long', candle)
            
            # Sell signal: Id_up decreases and Id_down increases
            elif id_up_decreased and id_down_increased:
                st.write("SELL SIGNAL detected")
                self.enter_trade('short', candle)
        
        # Check for exit signals - MODIFIED SECTION
        elif self.current_position == 'long':
            # Exit long when a short signal occurs (Id_up decreases AND Id_down increases)
            if id_up_decreased and id_down_increased:
                st.write("EXIT LONG SIGNAL (opposite signal detected)")
                closed_trade = self.exit_trade(candle, 'opposite_signal')
        
        elif self.current_position == 'short':
            # Exit short when a long signal occurs (Id_down decreases AND Id_up increases)
            if id_down_decreased and id_up_increased:
                st.write("EXIT SHORT SIGNAL (opposite signal detected)")
                closed_trade = self.exit_trade(candle, 'opposite_signal')
        
        # Update previous values
        self.prev_id_up = current_id_up
        self.prev_id_down = current_id_down
        
        return closed_trade
    
    def enter_trade(self, direction: str, candle: Dict):
        """Enter a new trade"""
        self.current_position = direction
        self.entry_price = candle['close']
        self.entry_time = candle['timestamp']
        st.write(f"[Trade] Entered {direction} at {self.entry_price:.2f} on {self.entry_time}")
    
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
        st.write(f"[Trade] Exited {self.current_position} at {exit_price:.2f} on {candle['timestamp']} "
              f"(PnL: {pnl:.2f}%)")
        
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        
        return trade
    
    def run_backtest(self, data: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical data"""
        self.reset()  # Clear any existing state
        
        st.write("\n[Backtest] Running backtest...")
        st.write(f"Data columns: {data.columns.tolist()}")
        
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
            st.warning("[Backtest] No trades were generated")
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
        
        st.write("\n[Backtest] Results:")
        st.write(f"Total trades: {len(self.trades)}")
        st.write(f"Profitable trades: {profitable_trades}")
        st.write(f"Win rate: {win_rate:.2%}")
        st.write(f"Avg profit: {avg_profit:.2f}%")
        st.write(f"Max drawdown: {max_drawdown:.2f}%")
        st.write(f"Sharpe ratio: {sharpe_ratio:.2f}")
        
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
        st.write(f"[Analyzer] Initialized for {timeframe} timeframe with threshold {self.failure_threshold}%")

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
            
            # Print significant failures to terminal
            if failure_mag > 1.0:  # Only print failures > 1%
                st.write(f"[Failure] {candle.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                      f"Price moved against prediction by {failure_mag:.2f}% "
                      f"(Id_up: {Id_up:.1f}, Id_down: {Id_down:.1f})")
        
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
        
        st.write(f"\n[Stats] Current timeframe: {self.timeframe}")
        st.write(f"  Total candles analyzed: {stats['total_candles']}")
        st.write(f"  Failed predictions: {stats['failed_candles']} ({stats['failure_percentage']:.2f}%)")
        st.write(f"  Average failure magnitude: {stats['average_failure_magnitude']:.2f}%")
        if stats['recent_failures']:
             st.write(f"  Recent failures: {[f'{x:.2f}%' for x in stats['recent_failures']]}")
             
        return stats

class MarketDataFetcher:
    @staticmethod
    def fetch_data(ticker: str, period: str, interval: str) -> List[Candle]:
        """Fetch OHLC data from yfinance and convert to Candle objects"""
        st.write(f"\n[Data Fetch] Downloading {ticker} data for {period} period at {interval} interval...")
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
            
            st.write(f"[Data Fetch] Successfully fetched {len(candles)} candles")
            st.write(f"  Date range: {candles[0].timestamp} to {candles[-1].timestamp}")
            st.write(f"  Price range: ${min(c.close for c in candles):.2f}-${max(c.close for c in candles):.2f}")
            
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
        st.write("\n[Model] Initialized Divergence Model")
        st.write(f"  Default ticker: {self.ticker}")
        st.write(f"  Default period: {self.period}")
        st.write(f"  Default timeframe: {self.timeframe}")
    
    def setup_analyzers(self):
        """Initialize analyzers for all timeframes"""
        configs = self.get_configurations()
        self.analyzers = {
            tf: DivergenceAnalyzer(cfg, tf) 
            for tf, cfg in configs.items()
        }
        st.write("[Model] Initialized analyzers for timeframes:", list(configs.keys()))
    
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
        st.write(f"\n[Model] Starting processing for {ticker} ({timeframe}, {period})")
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
        
        st.write(f"[Model] Processing {len(candles)} candles...")
        for candle in candles:
            result = analyzer.process_candle(candle, prev_candle)
            results.append(result)
            prev_candle = candle
        
        self.current_results = pd.DataFrame(results)
        st.write(f"[Model] Completed processing {len(results)} candles")
        return self.current_results
    
    def run_trading_backtest(self) -> BacktestResult:
        """Run the trading algorithm on current data"""
        if self.current_results is None or self.current_results.empty:
            raise ValueError("No data available for backtesting")
        
        st.write("\n[Model] Starting backtest...")
        st.write(f"Columns available: {self.current_results.columns.tolist()}")
        
        # Verify required columns exist
        required_cols = ['timestamp', 'close', 'Id_up', 'Id_down']
        if not all(col in self.current_results.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.current_results.columns]
            raise ValueError(f"Missing required columns for backtest: {missing}")
        
        algorithm = TradingAlgorithm()
        result = algorithm.run_backtest(self.current_results)
        
        st.write("\n[Model] Backtest completed")
        st.write(f"Total trades: {result.total_trades}")
        return result
    
    def get_plot_data(self) -> Optional[Dict]:
        """Prepare data for plotting"""
        if self.current_results is None or self.current_results.empty:
            st.warning("[Model] Warning: No data available for plotting")
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
            st.warning(f"[Model] Warning: No analyzer found for timeframe {self.timeframe}")
            return None
        return self.analyzers[self.timeframe].get_stats_summary()

def main():
    st.set_page_config(layout="wide", page_title="Kalman Filter Divergence Analyzer")
    st.title("Kalman Filter Divergence Analyzer")
    
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = DivergenceModel()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL")
        timeframe = st.selectbox(
            "Timeframe",
            options=["1m", "5m", "15m", "1h", "1d", "1wk"],
            index=2
        )
        period = st.selectbox(
            "Period",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        if st.button("Process Data"):
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
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
        
        if st.button("Run Backtest", disabled=not st.session_state.get('data_loaded', False)):
            with st.spinner("Running backtest..."):
                try:
                    backtest_result = st.session_state.model.run_trading_backtest()
                    st.session_state.backtest_result = backtest_result
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Divergence Statistics")
        if 'stats' in st.session_state and st.session_state.stats:
            stats = st.session_state.stats
            st.metric("Total Candles Analyzed", stats['total_candles'])
            st.metric("Failed Predictions", f"{stats['failed_candles']} ({stats['failure_percentage']:.2f}%)")
            st.metric("Average Failure Magnitude", f"{stats['average_failure_magnitude']:.2f}%")
            
            if stats['recent_failures']:
                st.write("Recent Failures:")
                st.write([f"{x:.2f}%" for x in stats['recent_failures']])
        else:
            st.info("No statistics available. Process data first.")
    
    with col2:
        st.subheader("Backtest Results")
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result
            st.metric("Total Trades", result.total_trades)
            st.metric("Profitable Trades", result.profitable_trades)
            st.metric("Win Rate", f"{result.win_rate:.1%}")
            st.metric("Avg Profit", f"{result.avg_profit:.2f}%")
            st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%")
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        else:
            st.info("No backtest results available. Run backtest first.")
    
    # Plot area
    st.subheader("Analysis Plots")
    if 'plot_data' in st.session_state and st.session_state.plot_data:
        plot_data = st.session_state.plot_data
        
        # Create tabs for different plot views
        tab1, tab2 = st.tabs(["Price and Divergence", "Trade Analysis"])
        
        with tab1:
            fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig1.subplots_adjust(hspace=0.3)
            
            # Plot price with failure markers
            ax1.plot(plot_data['timestamp'], plot_data['close'], label='Price', color='black')
            
            # Mark failure points
            failure_mask = plot_data['is_failure']
            if np.any(failure_mask):
                failure_times = np.array(plot_data['timestamp'])[failure_mask]
                failure_prices = np.array(plot_data['close'])[failure_mask]
                ax1.scatter(failure_times, failure_prices, color='red', label='Divergence Failure', zorder=5)
            
            ax1.set_ylabel('Price')
            ax1.set_title(f"{plot_data['ticker']} {plot_data['timeframe']} - Last {plot_data['period']}")
            ax1.legend()
            ax1.grid(True)
            
            # Plot divergence indices
            ax2.plot(plot_data['timestamp'], plot_data['Id_up'], label='Id_up', color='green')
            ax2.plot(plot_data['timestamp'], plot_data['Id_down'], label='Id_down', color='red')
            ax2.axhline(95, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(5, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Divergence Index')
            ax2.legend()
            ax2.grid(True)
            
            # Plot failure magnitudes
            if np.any(failure_mask):
                failure_magnitudes = np.array(plot_data['failure_magnitude'])
                failure_magnitudes[~failure_mask] = np.nan
                ax3.bar(plot_data['timestamp'], failure_magnitudes, color='red', label='Failure Magnitude')
            
            ax3.set_ylabel('Failure Magnitude (%)')
            ax3.set_xlabel('Date')
            ax3.legend()
            ax3.grid(True)
            
            # Rotate x-axis labels
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            st.pyplot(fig1)
        
        with tab2:
            if 'backtest_result' in st.session_state:
                result = st.session_state.backtest_result
                
                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                fig2.subplots_adjust(hspace=0.3)
                
                # Plot price with trade markers
                ax1.plot(plot_data['timestamp'], plot_data['close'], label='Price', color='black')
                
                # Mark trade entries and exits
                for trade in result.trades:
                    entry_color = 'green' if trade.direction == 'long' else 'red'
                    exit_color = 'red' if trade.pnl and trade.pnl > 0 else 'green'
                    
                    ax1.scatter(
                        trade.entry_time,
                        trade.entry_price,
                        color=entry_color,
                        marker='^',
                        s=100,
                        label='Entry' if trade == result.trades[0] else ""
                    )
                    
                    if trade.exit_time:
                        ax1.scatter(
                            trade.exit_time,
                            trade.exit_price,
                            color=exit_color,
                            marker='v',
                            s=100,
                            label='Exit' if trade == result.trades[0] else ""
                        )
                
                ax1.set_ylabel('Price')
                ax1.set_title(f"{plot_data['ticker']} {plot_data['timeframe']} - Trade Analysis")
                ax1.legend()
                ax1.grid(True)
                
                # Plot divergence indices
                ax2.plot(plot_data['timestamp'], plot_data['Id_up'], label='Id_up', color='green')
                ax2.plot(plot_data['timestamp'], plot_data['Id_down'], label='Id_down', color='red')
                ax2.axhline(95, color='gray', linestyle='--', alpha=0.5)
                ax2.axhline(5, color='gray', linestyle='--', alpha=0.5)
                ax2.set_ylabel('Divergence Index')
                ax2.legend()
                ax2.grid(True)
                
                # Rotate x-axis labels
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                st.pyplot(fig2)
            else:
                st.info("Run backtest to see trade analysis")
    else:
        st.info("Process data to see plots")
    
    # Results table
    st.subheader("Recent Results")
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
            height=400
        )
    else:
        st.info("No results to display. Process data first.")

if __name__ == "__main__":
    main()
