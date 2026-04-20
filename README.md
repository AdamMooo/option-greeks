# Option Greeks Visualizer

Interactive Black-Scholes dashboard for exploring how option price and risk sensitivity evolve across spot, strike, and time.

## What It Does

Computes the full set of Greeks in real time across a configurable spot price range. Every parameter adjustment — implied volatility, days to expiration, risk-free rate, strike — immediately rerenders all six charts simultaneously. Current moneyness (ITM / ATM / OTM) and theoretical option price are displayed live at the input spot.

**Greeks computed:** Delta, Gamma, Theta, Vega, Rho, and option price.

## Implementation

- Closed-form Black-Scholes with correct call/put asymmetry for delta, theta, and rho
- Greeks evaluated at 100 evenly spaced points across a user-defined price window
- Strike displayed as a vertical reference line across all six subplots

## Stack

Python · Streamlit · Plotly · NumPy · SciPy
