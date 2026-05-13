import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import math

# Set page configuration
st.set_page_config(
    page_title="Options Greeks Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptionsGreeksCalculator:
    def __init__(self):
        pass
    
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price and Greeks
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        # Avoid division by zero
        if T <= 0 or sigma <= 0:
            return {
                'price': max(S - K, 0) if option_type == 'call' else max(K - S, 0),
                'delta': 1.0 if option_type == 'call' else -1.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        if option_type == 'call':
            # Call option
            price = S * N_d1 - K * np.exp(-r * T) * N_d2
            delta = N_d1
            theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * N_d2) / 365  # Per day
            rho = K * T * np.exp(-r * T) * N_d2 / 100  # Per 1% change
        else:
            # Put option
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = N_d1 - 1
            theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Per day
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change
        
        # Gamma and Vega are the same for calls and puts
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        vega = S * n_d1 * np.sqrt(T) / 100  # Per 1% change in volatility
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

def create_greeks_dataframe(strike, expiry_days, risk_free_rate, volatility, option_type, price_range):
    """Create a dataframe with Greeks calculations across different stock prices"""
    
    min_price, max_price = price_range
    stock_prices = np.linspace(min_price, max_price, 100)
    
    calculator = OptionsGreeksCalculator()
    T = expiry_days / 365.0  # Convert days to years
    
    results = []
    for S in stock_prices:
        greeks = calculator.black_scholes_greeks(S, strike, T, risk_free_rate, volatility, option_type)
        results.append({
            'stock_price': S,
            'option_price': greeks['price'],
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'rho': greeks['rho']
        })
    
    return pd.DataFrame(results)

def create_charts(df, strike_price):
    """Create interactive plotly charts for the Greeks"""
    
    # Create subplots with more spacing
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Option Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Option Price
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['option_price'],
            mode='lines',
            name='Option Price',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['delta'],
            mode='lines',
            name='Delta',
            line=dict(color='#ff7f0e', width=3)
        ),
        row=1, col=2
    )
    
    # Gamma
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['gamma'],
            mode='lines',
            name='Gamma',
            line=dict(color='#2ca02c', width=3)
        ),
        row=2, col=1
    )
    
    # Theta
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['theta'],
            mode='lines',
            name='Theta',
            line=dict(color='#d62728', width=3)
        ),
        row=2, col=2
    )
    
    # Vega
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['vega'],
            mode='lines',
            name='Vega',
            line=dict(color='#9467bd', width=3)
        ),
        row=3, col=1
    )
    
    # Rho
    fig.add_trace(
        go.Scatter(
            x=df['stock_price'], 
            y=df['rho'],
            mode='lines',
            name='Rho',
            line=dict(color='#8c564b', width=3)
        ),
        row=3, col=2
    )
    
    # Add vertical line at strike price
    for i in range(1, 4):
        for j in range(1, 3):
            fig.add_vline(
                x=strike_price, 
                line_dash="dash", 
                line_color="gray",
                opacity=0.7,
                row=i, col=j
            )
    
    # Update layout with more height and proper margins
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="Options Greeks Analysis",
        title_x=0.5,
        title_font_size=20,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Update x-axis labels with smaller font and proper positioning
    fig.update_xaxes(
        title_text="Stock Price ($)", 
        title_font_size=12,
        title_standoff=25
    )
    
    # Update y-axis labels with smaller font and proper positioning
    fig.update_yaxes(
        title_text="Option Price ($)", 
        row=1, col=1, 
        title_font_size=12,
        title_standoff=25
    )
    fig.update_yaxes(
        title_text="Delta", 
        row=1, col=2, 
        title_font_size=12,
        title_standoff=25
    )
    fig.update_yaxes(
        title_text="Gamma", 
        row=2, col=1, 
        title_font_size=12,
        title_standoff=25
    )
    fig.update_yaxes(
        title_text="Theta (per day)", 
        row=2, col=2, 
        title_font_size=12,
        title_standoff=25
    )
    fig.update_yaxes(
        title_text="Vega (per 1% vol)", 
        row=3, col=1, 
        title_font_size=12,
        title_standoff=25
    )
    fig.update_yaxes(
        title_text="Rho (per 1% rate)", 
        row=3, col=2, 
        title_font_size=12,
        title_standoff=25
    )
    
    return fig

def main():
    st.title("Options Greeks Interactive Visualizer")
    st.markdown("---")
    
    # Sidebar for parameters
    st.sidebar.header("Option Parameters")
    
    # Option type
    option_type = st.sidebar.selectbox(
        "Option Type",
        options=['call', 'put'],
        index=0
    )
    
    # Strike price
    strike_price = st.sidebar.number_input(
        "Strike Price ($)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0
    )
    
    # Days to expiration
    days_to_expiry = st.sidebar.slider(
        "Days to Expiration",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )
    
    # Volatility
    volatility = st.sidebar.slider(
        "Implied Volatility (%)",
        min_value=5.0,
        max_value=150.0,
        value=25.0,
        step=1.0
    ) / 100.0
    
    # Risk-free rate
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    ) / 100.0
    
    st.sidebar.markdown("---")
    st.sidebar.header("Chart Settings")
    
    # Price range
    price_min = st.sidebar.number_input(
        "Min Stock Price ($)",
        min_value=1.0,
        max_value=500.0,
        value=max(1.0, strike_price * 0.7),
        step=1.0
    )
    
    price_max = st.sidebar.number_input(
        "Max Stock Price ($)",
        min_value=price_min + 1,
        max_value=1000.0,
        value=strike_price * 1.3,
        step=1.0
    )
    
    # Current stock price for calculations
    current_stock_price = st.sidebar.number_input(
        "Current Stock Price ($)",
        min_value=1.0,
        max_value=1000.0,
        value=strike_price,
        step=0.01
    )
    
    # Calculate current Greeks
    calculator = OptionsGreeksCalculator()
    current_greeks = calculator.black_scholes_greeks(
        current_stock_price, strike_price, days_to_expiry/365.0, 
        risk_free_rate, volatility, option_type
    )
    
    # Display current option metrics
    st.header("Current Option Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Option Price", f"${current_greeks['price']:.2f}")
        st.metric("Delta", f"{current_greeks['delta']:.3f}")
    
    with col2:
        st.metric("Gamma", f"{current_greeks['gamma']:.3f}")
        st.metric("Theta", f"{current_greeks['theta']:.3f}")
    
    with col3:
        st.metric("Vega", f"{current_greeks['vega']:.3f}")
        st.metric("Rho", f"{current_greeks['rho']:.3f}")
    
    with col4:
        moneyness = current_stock_price / strike_price
        if moneyness > 1.02:
            money_status = "ITM" if option_type == 'call' else "OTM"
        elif moneyness < 0.98:
            money_status = "OTM" if option_type == 'call' else "ITM"
        else:
            money_status = "ATM"
        
        st.metric("Moneyness", f"{moneyness:.3f}")
        st.metric("Status", money_status)
    
    # Generate data and create charts
    df = create_greeks_dataframe(
        strike_price, days_to_expiry, risk_free_rate, 
        volatility, option_type, (price_min, price_max)
    )
    
    # Display charts
    st.header("Greeks Visualization")
    fig = create_charts(df, strike_price)
    st.plotly_chart(fig, use_container_width=True)
    
    # Greeks explanation
    st.header("Understanding the Greeks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delta")
        st.write("Measures how much the option price changes for a $1 change in the underlying stock price.")
        st.write("• Call options: 0 to 1")
        st.write("• Put options: -1 to 0")
        
        st.subheader("Gamma")
        st.write("Measures how much delta changes for a $1 change in the underlying stock price.")
        st.write("• Higher for at-the-money options")
        st.write("• Peaks near expiration")
        
        st.subheader("Theta")
        st.write("Measures how much the option price decreases each day (time decay).")
        st.write("• Usually negative (options lose value over time)")
        st.write("• Accelerates as expiration approaches")
    
    with col2:
        st.subheader("Vega")
        st.write("Measures how much the option price changes for a 1% change in implied volatility.")
        st.write("• Higher for at-the-money options")
        st.write("• Decreases as expiration approaches")
        
        st.subheader("Rho")
        st.write("Measures how much the option price changes for a 1% change in interest rates.")
        st.write("• Usually small impact")
        st.write("• More significant for longer-term options")
        
    # Data table (optional)
    if st.checkbox("Show Data Table"):
        st.subheader("Raw Data")
        st.dataframe(df.round(4))

if __name__ == "__main__":
    main()
