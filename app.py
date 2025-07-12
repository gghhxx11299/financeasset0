import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from scipy.stats import norm, percentileofscore
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
import warnings
from matplotlib.ticker import FuncFormatter

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom CSS for the entire application
st.markdown("""
<style>
    .welcome-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .option-button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        margin: 0.5rem;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
    }
    .option-button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #343a40;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header {
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .back-button {
        background-color: #6c757d;
        color: white;
        margin-top: 20px;
    }
    .back-button:hover {
        background-color: #5a6268;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div>div>select {
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .stExpander>div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'

# --- Sector ETFs ---
SECTOR_MAP = {
    "technology": ["XLK", "VGT", "QTEC"],
    "financial": ["XLF", "VFH", "IYF"],
    "energy": ["XLE", "VDE", "IXC"],
    "healthcare": ["XLV", "IBB", "VHT"],
    "consumer discretionary": ["XLY", "VCR", "FDIS"],
    "consumer staples": ["XLP", "VDC", "FSTA"],
    "industrial": ["XLI", "VIS", "VMI"],
    "utilities": ["XLU", "VPU"],
    "real estate": ["XLRE", "VNQ", "SCHH"],
    "materials": ["XLB", "VAW"],
    "agriculture": ["DBA", "COW", "MOO"],
    "gold": ["GLD", "IAU", "SGOL"],
    "oil": ["USO", "OIH", "XOP"],
    "cryptocurrency": ["BTC-USD", "ETH-USD", "GBTC"],
    "bonds": ["AGG", "BND", "LQD"],
    "semiconductors": ["SMH", "SOXX"],
    "retail": ["XRT", "RTH"],
    "telecommunications": ["XTL", "VOX"],
    "transportation": ["IYT", "XTN"],
}

# --- Pricing Models ---
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes option price"""
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree_price(S, K, T, r, sigma, option_type="call", steps=100):
    """Calculate option price using binomial tree model"""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if option_type == "call":
        values = [max(0, price - K) for price in prices]
    else:
        values = [max(0, K - price) for price in prices]

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            values[j] = (p * values[j + 1] + (1 - p) * values[j]) * math.exp(-r * dt)

    return values[0]

def monte_carlo_price(S, K, T, r, sigma, option_type="call", simulations=10000):
    """Calculate option price using Monte Carlo simulation"""
    np.random.seed(42)
    dt = T
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(simulations))
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return float(price)

# --- Greeks Calculations ---
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes Greeks"""
    if T <= 0 or sigma == 0:
        return dict(Delta=0, Gamma=0, Vega=0, Theta=0, Rho=0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_call if option_type == "call" else theta_put
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    rho = rho_call if option_type == "call" else rho_put

    return dict(
        Delta=float(delta),
        Gamma=float(gamma),
        Vega=float(vega),
        Theta=float(theta),
        Rho=float(rho)
    )

# --- Implied Volatility ---
def implied_volatility(option_market_price, S, K, T, r, option_type="call", tol=1e-5, max_iter=100):
    """Calculate implied volatility using bisection method"""
    sigma_low, sigma_high = 0.0001, 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = black_scholes_price(S, K, T, r, sigma_mid, option_type)
        if abs(price - option_market_price) < tol:
            return sigma_mid
        if price > option_market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return None

# --- Market Data Functions ---
def get_option_market_price(ticker, option_type, strike, expiry_date):
    """Fetch current market price for given option"""
    stock = yf.Ticker(ticker)
    try:
        if expiry_date not in stock.options:
            return None
        opt_chain = stock.option_chain(expiry_date)
        options = opt_chain.calls if option_type == "call" else opt_chain.puts
        row = options[options['strike'] == strike]
        return None if row.empty else float(row.iloc[0]['lastPrice'])
    except:
        return None

def get_us_10yr_treasury_yield():
    """Fetch current 10-year Treasury yield"""
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/Datasets/yield.csv"
    fallback_yield = 0.025  # fallback 2.5%

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        df = df.dropna(subset=["10 Yr"])
        if df.empty:
            return fallback_yield

        latest_yield_str = df["10 Yr"].iloc[-1]
        return float(latest_yield_str) / 100
    except Exception:
        return fallback_yield

# --- Volatility Analysis ---
def calculate_iv_percentile(ticker, current_iv, lookback_days=365):
    """Calculate how current IV compares to historical levels"""
    try:
        hist = yf.download(ticker, period=f"{lookback_days}d")["Close"]
        daily_returns = hist.pct_change().dropna()
        realized_vol = daily_returns.std() * np.sqrt(252)
        
        return float(percentileofscore([realized_vol], current_iv))
    except Exception as e:
        st.warning(f"Could not calculate IV percentile: {e}")
        return None

def plot_stock_volume(ticker, days_to_expiry):
    """Plot stock/ETF trading volume - handles both simple and MultiIndex DataFrames"""
    try:
        # 1. Fetch data with multiple fallback attempts
        stock_data = None
        fetch_attempts = [
            {'auto_adjust': True, 'actions': False},
            {'auto_adjust': False, 'actions': False},
            {'auto_adjust': True, 'actions': True},
            {'auto_adjust': False, 'actions': True}
        ]
        
        for attempt in fetch_attempts:
            try:
                stock_data = yf.download(
                    ticker,
                    period=f"{min(days_to_expiry, 365)}d",
                    progress=False,
                    **attempt
                )
                if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
                    break
            except:
                continue

        # 2. Validate data structure
        if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
            st.warning(f"⚠️ No market data available for {ticker}")
            return None

        # 3. Find volume column (handles both regular and MultiIndex columns)
        volume_col = None
        for col in stock_data.columns:
            # Case 1: Simple string column name (e.g., 'Volume')
            if isinstance(col, str) and 'volume' in col.lower():
                volume_col = col
                break
            # Case 2: MultiIndex tuple column (e.g., ('Volume', 'SPY'))
            elif isinstance(col, tuple) and any('volume' in str(s).lower() for s in col):
                volume_col = col
                break

        if not volume_col:
            st.warning(f"📊 Volume data missing for {ticker} (Available columns: {stock_data.columns.tolist()})")
            return None

        # 4. Clean data
        clean_data = stock_data[[volume_col]].dropna()
        if clean_data.empty:
            st.warning(f"🧹 No valid volume data after cleaning for {ticker}")
            return None

        # 5. Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clean_data.index,
            y=clean_data[volume_col],
            marker_color='#1f77b4',
            hovertemplate="<b>Date</b>: %{x|%b %d}<br><b>Volume</b>: %{y:,}<extra></extra>"
        ))
        
        # Add average line if we have valid data
        try:
            avg_volume = clean_data[volume_col].mean()
            fig.add_shape(
                type="line",
                x0=clean_data.index[0],
                x1=clean_data.index[-1],
                y0=avg_volume,
                y1=avg_volume,
                line=dict(color='#ff7f0e', dash='dot')
            )
            fig.add_annotation(
                x=clean_data.index[-1],
                y=avg_volume,
                text=f"Avg: {avg_volume:,.0f}",
                showarrow=False,
                xanchor='right',
                yanchor='bottom',
                font=dict(color="#ff7f0e")
            )
        except:
            pass
        
        fig.update_layout(
            title=f"<b>{ticker} Volume</b> | Last {len(clean_data)} Trading Days",
            yaxis_title="Shares Traded",
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {str(e)}")
        return None

def plot_black_scholes_sensitivities(S, K, T, r, sigma, option_type):
    """Create enhanced interactive sensitivity plot for Black-Scholes model"""
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=("Price vs Underlying Asset", 
                                      "Price vs Days to Expiry", 
                                      "Price vs Volatility"))
    
    S_range = np.linspace(0.5*S, 1.5*S, 100)
    prices_S = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]
    fig.add_trace(go.Scatter(
        x=S_range, 
        y=prices_S, 
        name='Price vs Underlying',
        line=dict(color='#636EFA'),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.1)',
        hovertemplate="<b>Stock Price</b>: $%{x:.2f}<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    T_range = np.linspace(0.01, T*2, 100)
    prices_T = [black_scholes_price(S, K, t, r, sigma, option_type) for t in T_range]
    fig.add_trace(go.Scatter(
        x=T_range*365, 
        y=prices_T, 
        name='Price vs Days to Expiry',
        line=dict(color='#EF553B'),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.1)',
        hovertemplate="<b>Days to Expiry</b>: %{x:.0f}<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=2, col=1)
    
    sigma_range = np.linspace(0.01, 2*sigma, 100)
    prices_sigma = [black_scholes_price(S, K, T, r, s, option_type) for s in sigma_range]
    fig.add_trace(go.Scatter(
        x=sigma_range*100, 
        y=prices_sigma, 
        name='Price vs Volatility',
        line=dict(color='#00CC96'),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)',
        hovertemplate="<b>Volatility</b>: %{x:.2f}%<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=3, col=1)
    
    fig.add_vline(x=S, row=1, col=1, line=dict(color='#636EFA', dash='dash'), 
                annotation_text=f'Current Price: ${S:.2f}',
                annotation_position="top right")
    
    fig.add_vline(x=T*365, row=2, col=1, line=dict(color='#EF553B', dash='dash'), 
                annotation_text=f'Current DTE: {T*365:.0f} days',
                annotation_position="top right")
    
    fig.add_vline(x=sigma*100, row=3, col=1, line=dict(color='#00CC96', dash='dash'), 
                annotation_text=f'Current IV: {sigma*100:.2f}%',
                annotation_position="top right")
    
    fig.update_layout(
        title=f'<b>Black-Scholes Sensitivities ({option_type.capitalize()} Option)</b>',
        height=900,
        showlegend=False,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=50, r=50, b=50, t=100),
        title_font=dict(size=20, color='#2c3e50'),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=3, col=1)
    
    fig.update_xaxes(title_text="Underlying Asset Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Days to Expiration", row=2, col=1)
    fig.update_xaxes(title_text="Implied Volatility (%)", row=3, col=1)
    
    return fig

# --- Reporting ---
def prepare_export_csv(greeks_df, summary_df, trading_advice):
    greeks_export = greeks_df.rename(columns={"Greek": "Metric"})
    summary_export = summary_df
    advice_export = trading_advice.rename(columns={"Advice": "Metric", "Reason": "Value"})
    
    export_df = pd.concat([greeks_export, summary_export, advice_export], ignore_index=True)
    return export_df.to_csv(index=False).encode('utf-8')

def generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital):
    """Generate personalized trading advice based on analysis"""
    advice = []
    reasons = []
    
    high_iv_divergence = any(d > 0.1 for d in iv_divergences.values())
    if high_iv_divergence:
        max_divergence = max(iv_divergences.values())
        advice.append("Reduce position size")
        reasons.append(f"High IV divergence ({max_divergence:.2f} > 0.1) suggests overpriced options")
    
    extreme_z = abs(latest_z) > 2
    if extreme_z:
        advice.append("Exercise caution")
        reasons.append(f"Extreme price movement (Z-score: {latest_z:.2f}) indicates potential mean reversion")
    
    low_correlation = correlation < 0.5
    if low_correlation:
        advice.append("Consider hedging")
        reasons.append(f"Low sector correlation ({correlation:.2f}) reduces hedging effectiveness")
    
    capital_ratio = capital / comfortable_capital
    if capital_ratio < 0.7:
        advice.append("Reduce trade size significantly")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount")
    elif capital_ratio < 0.9:
        advice.append("Reduce trade size moderately")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount")
    
    if not advice:
        advice.append("Normal trading conditions")
        reasons.append("All metrics within normal ranges - standard position sizing appropriate")
    
    return pd.DataFrame({
        "Advice": advice,
        "Reason": reasons
    })

# --- Portfolio Optimization Helpers ---
def calculate_covariance_matrix(returns):
    """Calculate covariance matrix with shrinkage"""
    n = len(returns)
    sample_cov = returns.cov()
    prior = np.diag(np.diag(sample_cov))
    shrinkage = 0.5
    return shrinkage * prior + (1 - shrinkage) * sample_cov

def calculate_expected_returns(returns, method='mean'):
    """Calculate expected returns using different methods"""
    if method == 'mean':
        return returns.mean() * 252
    elif method == 'capm':
        market_returns = returns.mean(axis=1)
        betas = returns.apply(lambda x: np.cov(x, market_returns)[0, 1] / np.var(market_returns))
        risk_free_rate = 0.025
        market_return = market_returns.mean() * 252
        return risk_free_rate + betas * (market_return - risk_free_rate)
    return returns.mean() * 252

def portfolio_performance(weights, returns, cov_matrix):
    """Calculate portfolio performance metrics"""
    port_return = np.sum(weights * calculate_expected_returns(returns))
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def optimize_portfolio(returns, method='mpt', target_return=None):
    """Portfolio optimization implementation"""
    cov_matrix = calculate_covariance_matrix(returns)
    expected_returns = calculate_expected_returns(returns)
    n_assets = len(expected_returns)
    
    if method == 'mpt':
        # Mean-variance optimization
        def objective(weights):
            port_return = np.sum(weights * expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_vol if port_vol > 0 else 0
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        if target_return:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
            )
        bounds = [(0, 1) for _ in range(n_assets)]
        
        result = minimize(
            objective,
            x0=np.array([1/n_assets]*n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        perf = portfolio_performance(weights, returns, cov_matrix)
        return weights, perf
    
    elif method == 'hrp':
        # Hierarchical Risk Parity (simplified)
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist.values, 0)
        
        # Hierarchical clustering
        link = linkage(squareform(dist), 'single')
        order = leaves_list(link)
        
        # Inverse variance allocation
        iv = 1 / np.diag(cov_matrix)
        weights = iv / np.sum(iv)
        weights = weights[order]
        
        perf = portfolio_performance(weights, returns, cov_matrix)
        return weights, perf
    
    else:
        # Equal weighting as fallback
        weights = np.array([1/n_assets]*n_assets)
        perf = portfolio_performance(weights, returns, cov_matrix)
        return weights, perf

# --- Options Pricing Module ---
def options_pricing_main():
    st.title("Options Profit & Capital Advisor")

    # Initialize session state variables
    if "calculation_done" not in st.session_state:
        st.session_state.calculation_done = False
    if "export_csv" not in st.session_state:
        st.session_state.export_csv = None
    if "greeks_df" not in st.session_state:
        st.session_state.greeks_df = None
    if "summary_info" not in st.session_state:
        st.session_state.summary_info = None
    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    if "trading_advice" not in st.session_state:
        st.session_state.trading_advice = None
    if "bs_sensitivities_fig" not in st.session_state:
        st.session_state.bs_sensitivities_fig = None
    if "iv_percentile" not in st.session_state:
        st.session_state.iv_percentile = None
    if "volume_fig" not in st.session_state:
        st.session_state.volume_fig = None

    # Input widgets
    st.markdown("### Input Parameters")
    with st.expander("Configure your option trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
            option_type = st.selectbox("Option Type", ["call", "put"])
            strike_price = st.number_input("Strike Price", min_value=0.0, value=150.0)
            days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
            risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.025)
            sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
            
        with col2:
            return_type = st.selectbox("Return Type", ["Simple", "Log"])
            comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
            max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
            min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
            pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

    # Calculation button
    st.markdown("---")
    calculate_clicked = st.button("Calculate Profit & Advice", key="calculate")

    # When Calculate button is pressed
    if calculate_clicked:
        with st.spinner("Calculating option values and generating advice..."):
            try:
                # Store input data
                st.session_state.input_data = {
                    "Stock Ticker": ticker,
                    "Option Type": option_type,
                    "Strike Price": strike_price,
                    "Days to Expiry": days_to_expiry,
                    "Risk-Free Rate": risk_free_rate,
                    "Sector": sector,
                    "Return Type": return_type,
                    "Comfortable Capital": comfortable_capital,
                    "Max Capital": max_capital,
                    "Min Capital": min_capital,
                    "Pricing Model": pricing_model
                }

                # Fetch live treasury yield
                live_rate = get_us_10yr_treasury_yield()
                if live_rate is not None:
                    risk_free_rate = live_rate

                T = days_to_expiry / 365
                stock_data = yf.Ticker(ticker).history(period="1d")
                if stock_data.empty:
                    st.error("Could not fetch stock data. Please check the ticker symbol.")
                    st.session_state.calculation_done = False
                    return
                
                S = float(stock_data["Close"].iloc[-1])

                # Find closest expiry date
                options_expiries = yf.Ticker(ticker).options
                expiry_date = None
                for date in options_expiries:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    diff_days = abs((dt - datetime.now()).days - days_to_expiry)
                    if diff_days <= 5:
                        expiry_date = date
                        break

                if expiry_date is None:
                    st.error("No matching expiry date found near the specified days to expiry.")
                    st.session_state.calculation_done = False
                    return

                # Get market price and implied volatility
                price_market = get_option_market_price(ticker, option_type, strike_price, expiry_date)
                if price_market is None:
                    st.error("Failed to fetch option market price. Try a closer-to-the-money strike.")
                    st.session_state.calculation_done = False
                    return

                iv = implied_volatility(price_market, S, strike_price, T, risk_free_rate, option_type)
                if iv is None:
                    st.error("Could not compute implied volatility. Try a closer-to-the-money strike.")
                    st.session_state.calculation_done = False
                    return

                # Calculate Greeks
                greeks = black_scholes_greeks(S, strike_price, T, risk_free_rate, iv, option_type)
                greeks_df = pd.DataFrame({
                    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                    "Value": [
                        greeks['Delta'],
                        greeks['Gamma'],
                        greeks['Vega'],
                        greeks['Theta'],
                        greeks['Rho']
                    ]
                })
                st.session_state.greeks_df = greeks_df

                # Calculate option price using selected model
                start = time.time()
                if pricing_model == "Black-Scholes":
                    price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
                elif pricing_model == "Binomial Tree":
                    price = binomial_tree_price(S, strike_price, T, risk_free_rate, iv, option_type)
                elif pricing_model == "Monte Carlo":
                    price = monte_carlo_price(S, strike_price, T, risk_free_rate, iv, option_type)
                else:
                    price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
                end = time.time()
                calc_time = end - start

                # Sector analysis
                etfs = SECTOR_MAP.get(sector, [])
                symbols = [ticker] + etfs
                df = yf.download(symbols, period="1mo", interval="1d")["Close"].dropna(axis=1, how="any")

                if return_type == "Log":
                    returns = (df / df.shift(1)).apply(np.log).dropna()
                else:
                    returns = df.pct_change().dropna()

                # Z-score calculation
                window = 20
                zscore = ((df[ticker] - df[ticker].rolling(window).mean()) / df[ticker].rolling(window).std()).dropna()
                latest_z = float(zscore.iloc[-1]) if not zscore.empty else 0

                # Correlation analysis
                correlation = float(returns.corr().loc[ticker].drop(ticker).mean())
                iv_divergences = {etf: iv - 0.2 for etf in df.columns if etf != ticker}

                # Capital adjustment logic
                capital = comfortable_capital
                if any(d > 0.1 for d in iv_divergences.values()):
                    capital *= 0.6
                if abs(latest_z) > 2:
                    capital *= 0.7
                if correlation < 0.5:
                    capital *= 0.8

                capital = max(min_capital, min(max_capital, capital))

                # IV percentile analysis
                iv_percentile = calculate_iv_percentile(ticker, iv)
                st.session_state.iv_percentile = iv_percentile
                
                # Generate stock volume chart
                volume_fig = plot_stock_volume(ticker, days_to_expiry)
                st.session_state.volume_fig = volume_fig

                # Generate trading advice
                trading_advice = generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital)
                
                # Add warning if IV is extreme
                if iv_percentile and iv_percentile > 90:
                    trading_advice = pd.concat([
                        trading_advice,
                        pd.DataFrame({
                            "Advice": ["Market Stress Warning"],
                            "Reason": [f"IV is in top {100-iv_percentile:.0f}% of historical levels - possible crisis ahead"]
                        })
                    ])
                
                st.session_state.trading_advice = trading_advice

                # Prepare summary DataFrame
                summary_df = pd.DataFrame({
                    "Metric": ["Market Price", f"Model Price ({pricing_model})", "Implied Volatility (IV)", "Suggested Capital", "Calculation Time"],
                    "Value": [
                        f"${price_market:.2f}",
                        f"${float(price):.2f}",
                        f"{iv*100:.2f}%",
                        f"${float(capital):.2f}",
                        f"{float(calc_time):.4f} seconds"
                    ]
                })
                st.session_state.summary_info = summary_df

                # Prepare CSV export
                csv = prepare_export_csv(greeks_df, summary_df, trading_advice)
                st.session_state.export_csv = csv

                # Profit vs capital plot
                capitals = list(range(int(min_capital), int(max_capital) + 1, 100))
                profits = []
                profits_ci_lower = []
                profits_ci_upper = []

                if pricing_model == "Monte Carlo":
                    simulations = 10000
                    np.random.seed(42)
                    dt = T
                    ST = S * np.exp((risk_free_rate - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * np.random.randn(simulations))
                    if option_type == "call":
                        payoffs = np.maximum(ST - strike_price, 0)
                    else:
                        payoffs = np.maximum(strike_price - ST, 0)
                    discounted_payoffs = np.exp(-risk_free_rate * T) * payoffs
                    price_samples = discounted_payoffs

                    for cap in capitals:
                        contracts = int(cap / (price * 100)) if price > 0 else 0
                        profits_samples = contracts * 100 * (price_samples * 1.05 - price_samples)
                        mean_profit = float(profits_samples.mean())
                        std_profit = float(profits_samples.std())
                        ci_lower = mean_profit - 1.96 * std_profit / np.sqrt(simulations)
                        ci_upper = mean_profit + 1.96 * std_profit / np.sqrt(simulations)
                        profits.append(mean_profit)
                        profits_ci_lower.append(ci_lower)
                        profits_ci_upper.append(ci_upper)
                else:
                    for cap in capitals:
                        contracts = int(cap / (price * 100)) if price > 0 else 0
                        profit = contracts * 100 * (price * 1.05 - price)
                        profits.append(float(profit))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=capitals,
                    y=profits,
                    mode='lines+markers',
                    name='Expected Profit',
                    line=dict(color='#4CAF50', width=2),
                    marker=dict(size=8, color='#4CAF50'),
                    hovertemplate='<b>Capital</b>: $%{x:,.0f}<br><b>Profit</b>: $%{y:,.2f}<extra></extra>',
                ))

                if pricing_model == "Monte Carlo":
                    fig.add_trace(go.Scatter(
                        x=capitals + capitals[::-1],
                        y=profits_ci_upper + profits_ci_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(76, 175, 80, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Interval",
                    ))

                fig.update_layout(
                    title=f"<b>Expected Profit vs Capital for {ticker} {option_type.capitalize()} Option</b>",
                    xaxis_title="Capital Invested ($)",
                    yaxis_title="Expected Profit ($)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500,
                    margin=dict(l=50, r=50, b=50, t=80),
                    title_font=dict(size=18, color="#2c3e50"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.session_state.plot_fig = fig

                # Generate Black-Scholes sensitivities plot if using BS model
                if pricing_model == "Black-Scholes":
                    bs_sensitivities_fig = plot_black_scholes_sensitivities(S, strike_price, T, risk_free_rate, iv, option_type)
                    st.session_state.bs_sensitivities_fig = bs_sensitivities_fig

                st.session_state.calculation_done = True
                st.success("Calculation complete!")

            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                st.session_state.calculation_done = False

    # Display results if calculation is done
    if st.session_state.calculation_done:
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        # Metrics in cards
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Option Greeks")
                if st.session_state.greeks_df is not None:
                    st.dataframe(st.session_state.greeks_df, use_container_width=True)
            
            with col2:
                st.markdown("### Summary Metrics")
                if st.session_state.summary_info is not None:
                    st.dataframe(st.session_state.summary_info, use_container_width=True)
            
            with col3:
                if st.session_state.iv_percentile is not None:
                    st.markdown("### Volatility Context")
                    st.metric(
                        label="Implied Volatility Percentile",
                        value=f"{st.session_state.iv_percentile:.0f}th percentile",
                        help="How current IV compares to 1-year history (higher = more extreme)"
                    )
        
        # Trading Advice
        st.markdown("### Trading Advice")
        with st.expander("View detailed trading recommendations"):
            if st.session_state.trading_advice is not None:
                st.dataframe(st.session_state.trading_advice, use_container_width=True)
        
        # Plots
        if st.session_state.plot_fig is not None:
            st.plotly_chart(st.session_state.plot_fig, use_container_width=True)
        
        if st.session_state.volume_fig is not None:
            st.plotly_chart(st.session_state.volume_fig, use_container_width=True)
        
        if st.session_state.bs_sensitivities_fig is not None:
            st.plotly_chart(st.session_state.bs_sensitivities_fig, use_container_width=True)
        
        # Export buttons
        st.markdown("---")
        if st.session_state.export_csv is not None:
            st.download_button(
                label="Download CSV Report",
                data=st.session_state.export_csv,
                file_name="options_analysis_report.csv",
                mime="text/csv"
            )

# --- Portfolio Optimization Module ---
def portfolio_optimization_page():
    """Streamlit interface for portfolio optimization"""
    st.title("Portfolio Optimizer")
    
    # Input parameters
    with st.expander("Portfolio Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_area(
                "Asset Tickers (comma separated)",
                value="AAPL, MSFT, GOOG, AMZN, TSLA, JPM, JNJ, PG, WMT, DIS"
            )
            tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            
            method = st.selectbox(
                "Optimization Method",
                ["MPT (Modern Portfolio Theory)",
                 "HRP (Hierarchical Risk Parity)"]
            )
            
        with col2:
            end_date = datetime.now()
            start_date = st.date_input(
                "Historical Data Start Date",
                value=end_date - timedelta(days=5*365))
            
            if "MPT" in method:
                target_return = st.slider(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                ) / 100
    
    # Run optimization
    if st.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            try:
                # Download data
                data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                returns = data.pct_change().dropna()
                
                if len(tickers) == 1:
                    st.error("Please enter at least 2 tickers for portfolio optimization")
                    return
                
                # Run selected optimization method
                method_code = method.split(" ")[0].lower()
                weights, performance = optimize_portfolio(
                    returns,
                    method=method_code,
                    target_return=target_return if "MPT" in method else None
                )
                
                weights = pd.Series(weights, index=returns.columns)
                weights = weights[weights > 0.01]  # Filter out small weights
                weights = weights / weights.sum()  # Re-normalize
                
                # Display results
                st.success("Optimization completed successfully!")
                
                # Show allocation and metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Optimal Allocation")
                    alloc_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                    alloc_df['Weight'] = alloc_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(alloc_df.sort_values('Weight', ascending=False))
                
                with col2:
                    st.subheader("Key Metrics")
                    metrics = {
                        "Expected Return": f"{performance[0]*100:.2f}%",
                        "Annual Volatility": f"{performance[1]*100:.2f}%",
                        "Sharpe Ratio": f"{performance[2]:.2f}"
                    }
                    for name, value in metrics.items():
                        st.metric(name, value)
                
                # Create visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "bar"}]],
                    subplot_titles=("Portfolio Allocation", "Risk-Return Profile")
                )
                
                # Pie chart for allocation
                fig.add_trace(
                    go.Pie(
                        labels=weights.index,
                        values=weights.values,
                        name="Allocation"
                    ),
                    row=1, col=1
                )
                
                # Bar chart for metrics
                fig.add_trace(
                    go.Bar(
                        x=list(metrics.keys()),
                        y=[float(v.strip('%')) if '%' in v else float(v) for v in metrics.values()],
                        marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                        name="Metrics"
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    template="plotly_white",
                    title_text=f"{method} Portfolio Optimization Results",
                    margin=dict(l=50, r=50, b=50, t=100)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

# --- Fundamental Analysis Module ---
def fundamental_analysis_page():
    """Streamlit interface for fundamental analysis"""
    st.title("Fundamental Analyzer")
    st.markdown("**This tool is for stocks only** - ETF data may be incomplete")
    
    # Input parameters
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    
    if st.button("Analyze Fundamentals"):
        with st.spinner("Fetching data..."):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info:
                    st.error("Could not fetch data for this ticker")
                    return
                
                # Display key metrics
                st.subheader("Company Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sector", info.get('sector', 'N/A'))
                    st.metric("Industry", info.get('industry', 'N/A'))
                    st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A')
                
                with col2:
                    st.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    st.metric("Forward P/E", info.get('forwardPE', 'N/A'))
                    st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A')
                
                # Financial statements
                st.subheader("Financial Statements")
                
                try:
                    # Income Statement
                    income = stock.financials.T
                    if not income.empty:
                        st.markdown("**Income Statement**")
                        st.dataframe(income[['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']].style.format("${:,.0f}"))
                    
                    # Balance Sheet
                    balance = stock.balance_sheet.T
                    if not balance.empty:
                        st.markdown("**Balance Sheet**")
                        st.dataframe(balance[['Total Assets', 'Total Liabilities', 'Total Equity']].style.format("${:,.0f}"))
                    
                    # Cash Flow
                    cashflow = stock.cashflow.T
                    if not cashflow.empty:
                        st.markdown("**Cash Flow Statement**")
                        st.dataframe(cashflow[['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']].style.format("${:,.0f}"))
                
                except Exception as e:
                    st.warning(f"Could not fetch complete financial statements: {e}")
                
                # Historical data chart
                st.subheader("Historical Performance")
                hist = stock.history(period="5y")
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        name='Price',
                        line=dict(color='#1f77b4')
                    ))
                    fig.update_layout(
                        title=f"{ticker} Price History",
                        yaxis_title="Price ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# --- Main App Navigation ---
def show_welcome_page():
    st.markdown('<div class="welcome-header">Welcome to Your Trading Analytics Suite</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('Options Pricing & Analysis', key='options_btn', help='Analyze options strategies and get trading advice'):
            st.session_state.current_page = 'options'
    
    with col2:
        if st.button('Portfolio Optimization', key='portfolio_btn', help='Optimize your portfolio allocation'):
            st.session_state.current_page = 'portfolio'
    
    with col3:
        if st.button('Stock Fundamentals', key='fundamentals_btn', help='Analyze company fundamentals and earnings'):
            st.session_state.current_page = 'fundamentals'
    
    st.markdown("""
    <div style="margin-top: 3rem;">
        <h3>About This Application</h3>
        <p>This trading analytics platform provides:</p>
        <ul>
            <li><strong>Options Pricing:</strong> Black-Scholes, Binomial Tree, and Monte Carlo models with Greeks calculation</li>
            <li><strong>Portfolio Optimization:</strong> Modern Portfolio Theory (MPT) and Hierarchical Risk Parity (HRP)</li>
            <li><strong>Fundamental Analysis:</strong> Comprehensive stock analysis including financial statements</li>
        </ul>
        <p>Select an analysis module above to get started.</p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    if st.session_state.current_page == 'welcome':
        show_welcome_page()
    elif st.session_state.current_page == 'options':
        options_pricing_main()
    elif st.session_state.current_page == 'portfolio':
        portfolio_optimization_page()
    elif st.session_state.current_page == 'fundamentals':
        fundamental_analysis_page()
    
    # Add a back button on all pages except welcome
    if st.session_state.current_page != 'welcome':
        if st.button('← Back to Main Menu', key='back_btn'):
            st.session_state.current_page = 'welcome'

if __name__ == "__main__":
    main()
