import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from scipy.stats import norm, percentileofscore
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
import warnings

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

def generate_pdf_report(input_data, greeks_df, summary_df, trading_advice):
    """Generate PDF report using FPDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, "Options Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Input Parameters", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Greeks", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in greeks_df.iterrows():
        pdf.cell(200, 10, f"{row['Greek']}: {row['Value']}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in summary_df.iterrows():
        pdf.cell(200, 10, f"{row['Metric']}: {row['Value']}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Trading Advice", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in trading_advice.iterrows():
        # Handle special characters in advice
        text = f"{row['Advice']}: {row['Reason']}"
        try:
            pdf.multi_cell(200, 10, text.encode('latin-1', 'replace').decode('latin-1'))
        except:
            pdf.multi_cell(200, 10, "Trading advice (special characters omitted)")

    pdf.ln(10)
    pdf.set_font("Arial", 'I', size=10)
    pdf.cell(200, 10, "Note: Interactive plots are available in the web interface", ln=True)

    # Save to bytes with error handling
    try:
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except:
        try:
            return pdf.output(dest='S').encode('utf-8')
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            return None

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

# --- Options Pricing Module ---
def options_pricing_main():
    st.title("Options Profit & Capital Advisor")

    # Initialize session state variables
    if "calculation_done" not in st.session_state:
        st.session_state.calculation_done = False
    if "export_csv" not in st.session_state:
        st.session_state.export_csv = None
    if "export_pdf" not in st.session_state:
        st.session_state.export_pdf = None
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
                # Store input data for PDF report
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

                # Generate PDF report
                try:
                    pdf = generate_pdf_report(st.session_state.input_data, greeks_df, summary_df, trading_advice)
                    if pdf is not None:
                        st.session_state.export_pdf = pdf
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                    st.session_state.export_pdf = None
                
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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.export_csv is not None:
                st.download_button(
                    label="Download CSV Report",
                    data=st.session_state.export_csv,
                    file_name="options_analysis_report.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.session_state.export_pdf is not None:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.export_pdf,
                    file_name="options_analysis_report.pdf",
                    mime="application/pdf"
                )

# --- Portfolio Optimization Module ---
class InstitutionalPortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = self._get_risk_free_rate()
        self.market_returns = None
        self.asset_returns = None
        self.cov_matrix = None
        self.beta_values = None
        
    def _get_risk_free_rate(self):
        """Fetch 10-year Treasury yield as risk-free rate"""
        try:
            treasury = yf.Ticker("^TNX")
            rate = treasury.history(period="1d")['Close'].iloc[-1]/100
            return max(0.01, rate)  # Floor at 1%
        except:
            return 0.025  # Fallback rate
        
    def _download_asset_data(self, tickers, start_date, end_date):
        """Fetch institutional-grade market data with error handling"""
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data = data.dropna(axis=1, how='all')
        
        # Fill missing data using forward fill and interpolation
        data = data.ffill().bfill()
        for col in data.columns:
            data[col] = data[col].interpolate(method='time')
            
        return data.pct_change().dropna()
    
    def _download_market_data(self, benchmark='^GSPC'):
        """Download market benchmark data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years of data
        market_data = yf.download(benchmark, start=start_date, end=end_date)['Adj Close']
        return market_data.pct_change().dropna()
    
    def calculate_betas(self, asset_returns, market_returns):
        """Calculate institutional-quality betas with robust regression"""
        cov_matrix = np.cov(asset_returns.T, market_returns)
        market_variance = np.var(market_returns)
        betas = cov_matrix[-1, :-1] / market_variance
        return pd.Series(betas, index=asset_returns.columns)
    
    def capm_expected_returns(self, asset_returns, market_returns):
        """Enhanced CAPM with momentum and liquidity factors"""
        betas = self.calculate_betas(asset_returns, market_returns)
        avg_market_return = np.mean(market_returns) * 252
        momentum = asset_returns.rolling(63).mean().iloc[-1]  # 3-month momentum
        liquidity = asset_returns.rolling(21).std().iloc[-1]  # 1-month volatility
        
        # Fama-French inspired adjustment
        er = self.risk_free_rate + betas * (avg_market_return - self.risk_free_rate)
        er += 0.3 * momentum - 0.1 * liquidity  # Adjusted for momentum and liquidity
        return er
    
    def optimize_portfolio(self, method, asset_returns, **kwargs):
        """Institutional-grade portfolio optimization"""
        if method == "MPT":
            return self._mean_variance_optimization(asset_returns, **kwargs)
        elif method == "PMPT":
            return self._postmodern_optimization(asset_returns, **kwargs)
        elif method == "HRP":
            return self._hierarchical_risk_parity(asset_returns)
        else:
            raise ValueError("Invalid optimization method")
    
    def _mean_variance_optimization(self, asset_returns, target_return=None):
        """Modern Portfolio Theory optimization with Black-Litterman enhancements"""
        mu = expected_returns.capm_return(asset_returns)
        S = risk_models.CovarianceShrinkage(asset_returns).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S)
        if target_return:
            ef.efficient_return(target_return)
        else:
            ef.max_sharpe()
        
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        return weights, performance
    
    def _postmodern_optimization(self, asset_returns, target_sortino=0.5):
        """Post-Modern Portfolio Theory with downside risk focus"""
        # Calculate expected returns and covariance
        mu = expected_returns.ema_historical_return(asset_returns)
        S = risk_models.exp_cov(asset_returns)
        
        # Define PMPT objective function (Sortino ratio maximization)
        def sortino_ratio(weights, returns, rf_rate):
            portfolio_returns = np.dot(returns, weights)
            downside_returns = portfolio_returns[portfolio_returns < rf_rate/252]
            if len(downside_returns) == 0:
                return 1000  # Arbitrary large value
            downside_risk = np.std(downside_returns) * np.sqrt(252)
            excess_return = np.mean(portfolio_returns) * 252 - rf_rate
            return -excess_return / downside_risk  # Negative for minimization
        
        # Constraints
        n_assets = len(mu)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # No shorting
        )
        
        # Optimization
        res = minimize(
            sortino_ratio,
            x0=np.array([1/n_assets]*n_assets),
            args=(asset_returns.values, self.risk_free_rate),
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        weights = pd.Series(res.x, index=asset_returns.columns)
        performance = self._calculate_pmpt_performance(weights, asset_returns)
        return weights, performance
    
    def _hierarchical_risk_parity(self, asset_returns):
        """Hierarchical Risk Parity for robust allocation"""
        hrp = HRPOpt(asset_returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        
        # Calculate performance metrics
        perf = (
            hrp.portfolio_performance(),
            self._calculate_tail_risk(weights, asset_returns)
        )
        return weights, perf
    
    def _calculate_pmpt_performance(self, weights, returns):
        """Calculate PMPT-specific performance metrics"""
        portfolio_returns = np.dot(returns, weights)
        
        # Downside risk metrics
        downside_returns = portfolio_returns[portfolio_returns < self.risk_free_rate/252]
        downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        excess_return = np.mean(portfolio_returns) * 252 - self.risk_free_rate
        sortino = excess_return / downside_risk if downside_risk > 0 else 1000
        
        # Tail risk metrics
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        return {
            'expected_return': np.mean(portfolio_returns) * 252,
            'downside_risk': downside_risk,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_tail_risk(self, weights, returns):
        """Calculate advanced tail risk metrics"""
        portfolio_returns = np.dot(returns, weights)
        
        # Extreme value theory metrics
        sorted_returns = np.sort(portfolio_returns)
        n = len(sorted_returns)
        var_99 = sorted_returns[int(0.01 * n)] * np.sqrt(252)
        cvar_99 = sorted_returns[:int(0.01 * n)].mean() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'var_99': var_99,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown
        }
    
    def generate_report(self, weights, performance, method):
        """Generate institutional-quality PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, f"{method} Portfolio Optimization Report", ln=1, align='C')
        pdf.ln(10)
        
        # Portfolio Allocation
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Optimal Portfolio Allocation", ln=1)
        pdf.set_font("Arial", size=12)
        
        for asset, weight in weights.items():
            pdf.cell(200, 10, f"{asset}: {weight*100:.2f}%", ln=1)
        
        # Performance Metrics
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Performance Metrics", ln=1)
        pdf.set_font("Arial", size=12)
        
        if method == "MPT":
            pdf.cell(200, 10, f"Expected Annual Return: {performance[0]*100:.2f}%", ln=1)
            pdf.cell(200, 10, f"Annual Volatility: {performance[1]*100:.2f}%", ln=1)
            pdf.cell(200, 10, f"Sharpe Ratio: {performance[2]:.2f}", ln=1)
        elif method == "PMPT":
            pdf.cell(200, 10, f"Expected Annual Return: {performance['expected_return']*100:.2f}%", ln=1)
            pdf.cell(200, 10, f"Downside Risk: {performance['downside_risk']*100:.2f}%", ln=1)
            pdf.cell(200, 10, f"Sortino Ratio: {performance['sortino_ratio']:.2f}", ln=1)
            pdf.cell(200, 10, f"95% VaR: {performance['var_95']*100:.2f}%", ln=1)
            pdf.cell(200, 10, f"95% CVaR: {performance['cvar_95']*100:.2f}%", ln=1)
        
        # Tail Risk Section
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Tail Risk Metrics", ln=1)
        pdf.set_font("Arial", size=12)
        
        tail_risk = self._calculate_tail_risk(weights, self.asset_returns)
        pdf.cell(200, 10, f"99% VaR: {tail_risk['var_99']*100:.2f}%", ln=1)
        pdf.cell(200, 10, f"99% CVaR: {tail_risk['cvar_99']*100:.2f}%", ln=1)
        pdf.cell(200, 10, f"Maximum Drawdown: {tail_risk['max_drawdown']*100:.2f}%", ln=1)
        
        # Efficient Frontier Plot
        if method == "MPT":
            self._plot_efficient_frontier(pdf)
        
        return pdf.output(dest='S').encode('latin1')
    
    def _plot_efficient_frontier(self, pdf):
        """Generate efficient frontier plot"""
        plt.figure(figsize=(8, 5))
        ef = EfficientFrontier(
            expected_returns.capm_return(self.asset_returns),
            risk_models.CovarianceShrinkage(self.asset_returns).ledoit_wolf()
        )
        fig, ax = plt.subplots()
        ef.plot_efficient_frontier(ax=ax, show_assets=True)
        ax.set_title("Efficient Frontier")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        
        # Save plot to PDF
        plt.savefig("efficient_frontier.png")
        pdf.image("efficient_frontier.png", x=10, y=pdf.get_y(), w=180)
        pdf.ln(85)  # Adjust based on image height
    
    def plot_optimization_results(self, weights, performance, method):
        """Create interactive visualization of results"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"colspan": 2}, None],
            subplot_titles=(
                "Portfolio Allocation",
                "Risk-Return Profile",
                "Historical Performance"
            ),
            vertical_spacing=0.15
        )
        
        # Pie chart for allocation
        fig.add_trace(
            go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                name="Allocation"
            ),
            row=1, col=1
        )
        
        # Bar chart for risk metrics
        if method == "MPT":
            metrics = ["Return", "Volatility", "Sharpe Ratio"]
            values = [performance[0], performance[1], performance[2]]
        else:
            metrics = ["Return", "Downside Risk", "Sortino Ratio"]
            values = [
                performance['expected_return'],
                performance['downside_risk'],
                performance['sortino_ratio']
            ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                name="Metrics"
            ),
            row=1, col=2
        )
        
        # Historical performance line chart
        portfolio_returns = np.dot(self.asset_returns, list(weights.values()))
        cum_returns = np.cumprod(1 + portfolio_returns) - 1
        
        fig.add_trace(
            go.Scatter(
                x=self.asset_returns.index,
                y=cum_returns,
                mode='lines',
                name="Portfolio Growth",
                line=dict(color='#9b59b6', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text=f"{method} Portfolio Optimization Results",
            margin=dict(l=50, r=50, b=50, t=100)
        )
        
        return fig

def portfolio_optimization_page():
    """Streamlit interface for portfolio optimization"""
    st.title("Institutional Portfolio Optimizer")
    
    # Initialize optimizer
    optimizer = InstitutionalPortfolioOptimizer()
    
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
                 "PMPT (Post-Modern Portfolio Theory)",
                 "HRP (Hierarchical Risk Parity)"]
            )
            
        with col2:
            end_date = datetime.now()
            start_date = st.date_input(
                "Historical Data Start Date",
                value=end_date - timedelta(days=5*365)
            
            if "PMPT" in method:
                target_sortino = st.slider(
                    "Minimum Sortino Ratio Target",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.0,
                    step=0.1
                )
            elif "MPT" in method:
                target_return = st.slider(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                ) / 100
    
    # Run optimization
    if st.button("Run Optimization"):
        with st.spinner("Running institutional-grade optimization..."):
            try:
                # Download data
                asset_returns = optimizer._download_asset_data(
                    tickers,
                    start_date=start_date,
                    end_date=end_date
                )
                market_returns = optimizer._download_market_data()
                
                optimizer.asset_returns = asset_returns
                optimizer.market_returns = market_returns
                optimizer.cov_matrix = asset_returns.cov()
                optimizer.beta_values = optimizer.calculate_betas(asset_returns, market_returns)
                
                # Run selected optimization method
                method_code = method.split(" ")[0]
                if method_code == "PMPT":
                    weights, performance = optimizer.optimize_portfolio(
                        method_code,
                        asset_returns,
                        target_sortino=target_sortino
                    )
                elif method_code == "MPT":
                    weights, performance = optimizer.optimize_portfolio(
                        method_code,
                        asset_returns,
                        target_return=target_return
                    )
                else:
                    weights, performance = optimizer.optimize_portfolio(
                        method_code,
                        asset_returns
                    )
                
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
                    
                    if method_code == "MPT":
                        metrics = {
                            "Expected Return": f"{performance[0]*100:.2f}%",
                            "Annual Volatility": f"{performance[1]*100:.2f}%",
                            "Sharpe Ratio": f"{performance[2]:.2f}"
                        }
                    elif method_code == "PMPT":
                        metrics = {
                            "Expected Return": f"{performance['expected_return']*100:.2f}%",
                            "Downside Risk": f"{performance['downside_risk']*100:.2f}%",
                            "Sortino Ratio": f"{performance['sortino_ratio']:.2f}"
                        }
                    else:
                        metrics = {
                            "Expected Return": f"{performance[0]*100:.2f}%",
                            "Annual Volatility": f"{performance[1]*100:.2f}%",
                            "CVaR (95%)": f"{performance[2]['cvar_95']*100:.2f}%"
                        }
                    
                    for name, value in metrics.items():
                        st.metric(name, value)
                
                # Visualizations
                st.plotly_chart(
                    optimizer.plot_optimization_results(weights, performance, method_code),
                    use_container_width=True
                )
                
                # Generate PDF report
                pdf_report = optimizer.generate_report(weights, performance, method_code)
                st.download_button(
                    label="Download Full PDF Report",
                    data=pdf_report,
                    file_name=f"{method_code}_portfolio_report.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

# --- Fundamental Analysis Module ---
class InstitutionalFundamentalAnalyzer:
    def __init__(self):
        self.ticker = None
        self.stock = None
        self.financials = {
            'income': None,
            'balance': None,
            'cashflow': None
        }
        self.earnings = None
        self.dividends = None
        self.analyst_info = None
        self.quarterly_results = None
        
    def _millions_formatter(self, x, pos):
        """Formatter for matplotlib to display in millions"""
        return f'${x/1e6:.0f}M'
    
    def _download_institutional_data(self, ticker):
        """Download comprehensive institutional data with error handling"""
        try:
            self.ticker = ticker.upper()
            self.stock = yf.Ticker(ticker)
            
            # Get financial statements with error handling
            try:
                self.financials['income'] = self.stock.financials.T
                self.financials['balance'] = self.stock.balance_sheet.T
                self.financials['cashflow'] = self.stock.cashflow.T
            except:
                st.warning("Could not fetch complete financial statements. Some data may be missing.")
                
            # Get earnings data
            try:
                self.earnings = self.stock.earnings
                self.quarterly_results = self.stock.quarterly_earnings
            except:
                st.warning("Could not fetch earnings data.")
                
            # Get dividends
            try:
                self.dividends = self.stock.dividends
            except:
                st.warning("Could not fetch dividend history.")
                
            # Get analyst info
            try:
                self.analyst_info = {
                    'recommendations': self.stock.recommendations,
                    'upgrades_downgrades': self.stock.upgrades_downgrades,
                    'earnings_estimates': self.stock.earnings_estimates,
                    'revenue_estimates': self.stock.revenue_estimates
                }
            except:
                st.warning("Could not fetch analyst estimates and recommendations.")
                
            return True
        except Exception as e:
            st.error(f"Failed to download data for {ticker}: {str(e)}")
            return False
    
    def _clean_financials(self, df, statement_type):
        """Clean and standardize financial statements"""
        if df is None or df.empty:
            return None
            
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        
        # Common financial statement items
        common_items = {
            'income': ['total revenue', 'cost of revenue', 'gross profit', 
                      'operating expenses', 'operating income', 'net income'],
            'balance': ['total assets', 'total liabilities', 'total equity',
                       'cash and cash equivalents', 'total debt'],
            'cashflow': ['operating cash flow', 'investing cash flow', 
                        'financing cash flow', 'free cash flow']
        }
        
        # Filter for important line items
        if statement_type in common_items:
            df = df[[col for col in df.columns if any(item in col.lower() for item in common_items[statement_type])]]
        
        return df
    
    def analyze_fundamentals(self, ticker):
        """Comprehensive fundamental analysis"""
        if not self._download_institutional_data(ticker):
            return False
            
        # Clean financial statements
        self.financials['income'] = self._clean_financials(self.financials['income'], 'income')
        self.financials['balance'] = self._clean_financials(self.financials['balance'], 'balance')
        self.financials['cashflow'] = self._clean_financials(self.financials['cashflow'], 'cashflow')
        
        return True
    
    def create_fundamentals_report(self):
        """Generate institutional-quality PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, f"{self.ticker} Fundamental Analysis Report", ln=1, align='C')
        pdf.ln(10)
        
        # Company Overview
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Company Overview", ln=1)
        pdf.set_font("Arial", size=12)
        
        try:
            info = self.stock.info
            pdf.cell(200, 10, f"Sector: {info.get('sector', 'N/A')}", ln=1)
            pdf.cell(200, 10, f"Industry: {info.get('industry', 'N/A')}", ln=1)
            pdf.cell(200, 10, f"Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B", ln=1)
            pdf.cell(200, 10, f"Enterprise Value: ${info.get('enterpriseValue', 0)/1e9:.2f}B", ln=1)
            pdf.cell(200, 10, f"P/E Ratio: {info.get('trailingPE', 'N/A')}", ln=1)
        except:
            pdf.cell(200, 10, "Could not fetch company info", ln=1)
        
        # Income Statement Summary
        if self.financials['income'] is not None:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Income Statement Trends", ln=1)
            pdf.set_font("Arial", size=12)
            
            for col in ['total revenue', 'gross profit', 'operating income', 'net income']:
                if col in self.financials['income'].columns:
                    latest = self.financials['income'][col].iloc[-1]
                    prev = self.financials['income'][col].iloc[-2] if len(self.financials['income']) > 1 else None
                    growth = f"({(latest-prev)/prev*100:.1f}%)" if prev else ""
                    pdf.cell(200, 10, f"{col}: ${latest/1e6:.0f}M {growth}", ln=1)
            
            # Add income statement plot
            self._plot_financial_statement(self.financials['income'], "Income Statement")
            pdf.image("financial_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)
        
        # Balance Sheet Analysis
        if self.financials['balance'] is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Balance Sheet Analysis", ln=1)
            pdf.set_font("Arial", size=12)
            
            # Key metrics
            if 'total assets' in self.financials['balance'].columns and 'total liabilities' in self.financials['balance'].columns:
                assets = self.financials['balance']['total assets'].iloc[-1]
                liabilities = self.financials['balance']['total liabilities'].iloc[-1]
                equity = assets - liabilities
                debt_ratio = liabilities / assets
                
                pdf.cell(200, 10, f"Total Assets: ${assets/1e6:.0f}M", ln=1)
                pdf.cell(200, 10, f"Total Liabilities: ${liabilities/1e6:.0f}M", ln=1)
                pdf.cell(200, 10, f"Shareholders' Equity: ${equity/1e6:.0f}M", ln=1)
                pdf.cell(200, 10, f"Debt-to-Assets Ratio: {debt_ratio:.2f}", ln=1)
            
            # Add balance sheet plot
            self._plot_financial_statement(self.financials['balance'], "Balance Sheet")
            pdf.image("financial_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)
        
        # Cash Flow Analysis
        if self.financials['cashflow'] is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Cash Flow Analysis", ln=1)
            pdf.set_font("Arial", size=12)
            
            if 'operating cash flow' in self.financials['cashflow'].columns:
                ocf = self.financials['cashflow']['operating cash flow'].iloc[-1]
                pdf.cell(200, 10, f"Operating Cash Flow: ${ocf/1e6:.0f}M", ln=1)
                
            if 'free cash flow' in self.financials['cashflow'].columns:
                fcf = self.financials['cashflow']['free cash flow'].iloc[-1]
                pdf.cell(200, 10, f"Free Cash Flow: ${fcf/1e6:.0f}M", ln=1)
            
            # Add cash flow plot
            self._plot_financial_statement(self.financials['cashflow'], "Cash Flow Statement")
            pdf.image("financial_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)
        
        # Earnings Analysis
        if self.earnings is not None or self.quarterly_results is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Earnings Analysis", ln=1)
            pdf.set_font("Arial", size=12)
            
            if self.earnings is not None:
                latest_eps = self.earnings.iloc[-1]['Earnings']
                pdf.cell(200, 10, f"Latest Annual EPS: ${latest_eps:.2f}", ln=1)
                
            if self.quarterly_results is not None:
                latest_q_eps = self.quarterly_results.iloc[-1]['Earnings']
                pdf.cell(200, 10, f"Latest Quarterly EPS: ${latest_q_eps:.2f}", ln=1)
            
            # Add earnings plot
            self._plot_earnings()
            pdf.image("earnings_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)
        
        # Dividend Analysis
        if self.dividends is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Dividend Analysis", ln=1)
            pdf.set_font("Arial", size=12)
            
            if len(self.dividends) > 0:
                latest_div = self.dividends.iloc[-1]
                pdf.cell(200, 10, f"Latest Dividend: ${latest_div:.2f}", ln=1)
                
                # Calculate yield if we have price info
                try:
                    info = self.stock.info
                    if 'currentPrice' in info:
                        yield_pct = (latest_div / info['currentPrice']) * 100
                        pdf.cell(200, 10, f"Current Yield: {yield_pct:.2f}%", ln=1)
                except:
                    pass
            
            # Add dividend plot
            self._plot_dividends()
            pdf.image("dividend_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)
        
        # Analyst Estimates
        if self.analyst_info and self.analyst_info.get('earnings_estimates') is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Analyst Estimates", ln=1)
            pdf.set_font("Arial", size=12)
            
            estimates = self.analyst_info['earnings_estimates']
            current_year = estimates[estimates['Fiscal Period'] == '+1y']
            next_year = estimates[estimates['Fiscal Period'] == '+2y']
            
            if not current_year.empty:
                avg_estimate = current_year['Avg. Estimate'].iloc[0]
                high_estimate = current_year['High Estimate'].iloc[0]
                low_estimate = current_year['Low Estimate'].iloc[0]
                
                pdf.cell(200, 10, f"Current Year EPS Estimate: ${avg_estimate:.2f}", ln=1)
                pdf.cell(200, 10, f"Estimate Range: ${low_estimate:.2f} to ${high_estimate:.2f}", ln=1)
            
            # Add estimates plot
            self._plot_analyst_estimates()
            pdf.image("estimates_plot.png", x=10, y=pdf.get_y(), w=180)
        
        return pdf.output(dest='S').encode('latin1')
    
    def _plot_financial_statement(self, df, title):
        """Plot financial statement trends"""
        if df is None or df.empty:
            return
            
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Plot each line item
        for col in df.columns:
            if df[col].abs().max() > 0:  # Only plot if we have data
                plt.plot(df.index, df[col]/1e6, label=col, marker='o')
        
        # Formatting
        ax.yaxis.set_major_formatter(FuncFormatter(self._millions_formatter))
        plt.title(title)
        plt.ylabel('Amount ($M)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("financial_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_earnings(self):
        """Plot earnings trends"""
        if self.earnings is None and self.quarterly_results is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Annual earnings
        if self.earnings is not None:
            ax1.plot(self.earnings.index, self.earnings['Earnings'], 'b-o')
            ax1.set_title('Annual Earnings Per Share (EPS)')
            ax1.set_ylabel('EPS ($)')
            ax1.grid(True)
        
        # Quarterly earnings
        if self.quarterly_results is not None:
            ax2.plot(self.quarterly_results.index, self.quarterly_results['Earnings'], 'g-o')
            ax2.set_title('Quarterly Earnings Per Share (EPS)')
            ax2.set_ylabel('EPS ($)')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("earnings_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_dividends(self):
        """Plot dividend history"""
        if self.dividends is None or self.dividends.empty:
            return
            
        plt.figure(figsize=(10, 4))
        self.dividends.plot(marker='o')
        plt.title('Dividend History')
        plt.ylabel('Dividend per Share ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("dividend_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_analyst_estimates(self):
        """Plot analyst estimates"""
        if not self.analyst_info or self.analyst_info.get('earnings_estimates') is None:
            return
            
        estimates = self.analyst_info['earnings_estimates']
        
        plt.figure(figsize=(10, 4))
        plt.errorbar(
            estimates['Fiscal Period'],
            estimates['Avg. Estimate'],
            yerr=[estimates['Avg. Estimate']-estimates['Low Estimate'], 
                  estimates['High Estimate']-estimates['Avg. Estimate']],
            fmt='o',
            capsize=5
        )
        plt.title('Analyst EPS Estimates')
        plt.ylabel('EPS ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("estimates_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        if not any([self.financials['income'], self.earnings, self.dividends]):
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Financial Statement Trends",
                "Earnings History",
                "Dividend History"
            ),
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]]
        )
        
        # Financial Statements
        if self.financials['income'] is not None:
            for col in self.financials['income'].columns:
                fig.add_trace(
                    go.Bar(
                        x=self.financials['income'].index,
                        y=self.financials['income'][col]/1e6,
                        name=col,
                        visible='legendonly' if len(self.financials['income'].columns) > 3 else True
                    ),
                    row=1, col=1
                )
        
        # Earnings
        if self.earnings is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.earnings.index,
                    y=self.earnings['Earnings'],
                    name='Annual EPS',
                    mode='lines+markers',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
        
        if self.quarterly_results is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.quarterly_results.index,
                    y=self.quarterly_results['Earnings'],
                    name='Quarterly EPS',
                    mode='lines+markers',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Dividends
        if self.dividends is not None and not self.dividends.empty:
            fig.add_trace(
                go.Scatter(
                    x=self.dividends.index,
                    y=self.dividends,
                    name='Dividends',
                    mode='lines+markers',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"{self.ticker} Fundamental Analysis",
            margin=dict(l=50, r=50, b=50, t=100),
            hovermode="x unified"
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Amount ($M)", row=1, col=1)
        fig.update_yaxes(title_text="EPS ($)", row=2, col=1)
        fig.update_yaxes(title_text="Dividend ($)", row=3, col=1)
        
        return fig

def fundamental_analysis_page():
    """Streamlit interface for fundamental analysis"""
    st.title("Institutional Fundamental Analyzer")
    st.markdown("**This tool is for stocks only** - ETF data may be incomplete")
    
    # Initialize analyzer
    analyzer = InstitutionalFundamentalAnalyzer()
    
    # Input parameters
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    
    if st.button("Analyze Fundamentals"):
        with st.spinner("Fetching institutional-grade data..."):
            if analyzer.analyze_fundamentals(ticker):
                st.success("Analysis completed successfully!")
                
                # Display interactive dashboard
                dashboard = analyzer.create_interactive_dashboard()
                if dashboard:
                    st.plotly_chart(dashboard, use_container_width=True)
                
                # Display key metrics
                st.subheader("Key Financial Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                # Income Statement Metrics
                if analyzer.financials['income'] is not None:
                    with col1:
                        st.markdown("**Income Statement**")
                        if 'total revenue' in analyzer.financials['income'].columns:
                            revenue = analyzer.financials['income']['total revenue'].iloc[-1]/1e9
                            st.metric("Revenue", f"${revenue:.2f}B")
                        
                        if 'net income' in analyzer.financials['income'].columns:
                            net_income = analyzer.financials['income']['net income'].iloc[-1]/1e9
                            st.metric("Net Income", f"${net_income:.2f}B")
                
                # Balance Sheet Metrics
                if analyzer.financials['balance'] is not None:
                    with col2:
                        st.markdown("**Balance Sheet**")
                        if 'total assets' in analyzer.financials['balance'].columns:
                            assets = analyzer.financials['balance']['total assets'].iloc[-1]/1e9
                            st.metric("Total Assets", f"${assets:.2f}B")
                        
                        if 'total liabilities' in analyzer.financials['balance'].columns:
                            liabilities = analyzer.financials['balance']['total liabilities'].iloc[-1]/1e9
                            st.metric("Total Liabilities", f"${liabilities:.2f}B")
                
                # Cash Flow Metrics
                if analyzer.financials['cashflow'] is not None:
                    with col3:
                        st.markdown("**Cash Flow**")
                        if 'operating cash flow' in analyzer.financials['cashflow'].columns:
                            ocf = analyzer.financials['cashflow']['operating cash flow'].iloc[-1]/1e9
                            st.metric("Operating Cash Flow", f"${ocf:.2f}B")
                        
                        if 'free cash flow' in analyzer.financials['cashflow'].columns:
                            fcf = analyzer.financials['cashflow']['free cash flow'].iloc[-1]/1e9
                            st.metric("Free Cash Flow", f"${fcf:.2f}B")
                
                # Generate PDF report
                pdf_report = analyzer.create_fundamentals_report()
                st.download_button(
                    label="Download Full PDF Report",
                    data=pdf_report,
                    file_name=f"{ticker}_fundamental_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to analyze the stock. Please check the ticker symbol.")

# --- Main App Navigation ---
def show_welcome_page():
    st.markdown('<div class="welcome-header">Welcome to Your Trading Analytics Suite</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('Options Pricing & Analysis', key='options_btn', help='Analyze options strategies and get trading advice'):
            st.session_state.current_page = 'options'
    
    with col2:
        if st.button('Portfolio Optimization', key='portfolio_btn', help='Optimize your portfolio using CAPM, MPT, or PMPT'):
            st.session_state.current_page = 'portfolio'
    
    with col3:
        if st.button('Stock Fundamentals', key='fundamentals_btn', help='Analyze company fundamentals and earnings'):
            st.session_state.current_page = 'fundamentals'
    
    st.markdown("""
    <div style="margin-top: 3rem;">
        <h3>About This Application</h3>
        <p>This institutional-grade trading analytics platform provides:</p>
        <ul>
            <li><strong>Options Pricing:</strong> Black-Scholes, Binomial Tree, and Monte Carlo models with Greeks calculation</li>
            <li><strong>Portfolio Optimization:</strong> Modern Portfolio Theory (MPT), Post-Modern Portfolio Theory (PMPT), and Hierarchical Risk Parity (HRP)</li>
            <li><strong>Fundamental Analysis:</strong> Comprehensive stock analysis including financial statements, earnings, and dividends</li>
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
