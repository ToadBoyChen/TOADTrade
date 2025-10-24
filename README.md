# TOADTrade

Python-based trading algorithm that uses multiple technical indicators to generate trading signals and backtest strategies on historical stock data.

## Overview

TOADTrade combines seven different technical indicators to create weighted trading signals:
- **Moving Averages (MA)**: Short-term vs long-term trend analysis
- **Relative Strength Index (RSI)**: Momentum oscillator for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence for trend changes
- **Bollinger Bands**: Volatility-based support/resistance levels
- **On-Balance Volume (OBV)**: Volume-price relationship analysis
- **Average True Range (ATR)**: Volatility measurement
- **Stochastic Oscillator**: Momentum indicator comparing closing price to price range

## Features

1. **Real-time Data Fetching**: Uses yfinance to fetch current stock data
2. **Multiple Technical Indicators**: 7 different indicators with proper signal generation
3. **Weighted Signal System**: Combines all indicators for comprehensive 
4. **Performance Metrics**: Clear comparison of strategy vs market returns
5. **Forecasting**: SARIMAX-based price prediction (experimental)

## Usage

The program will:
1. Ask for a stock symbol (e.g., AAPL, TSLA, MSFT)
2. Ask for the analysis period (1-5 years)
3. Fetch historical data
4. Generate technical indicators and signals
5. Run backtesting simulation
6. Display performance comparison
7. Generate forecasts
8. Save results to files

### Risk Management
- **Tolerance Levels**: 1 (conservative) to 5 (very risky)
- **Position Sizing**: All-in/all-out approach (can be modified)
- **Transaction Costs**: Not currently included (can be added)

## Customization

### Modify Indicator Parameters
Edit the `parameters` dictionary in `main()`:
```python
parameters = {
    'shortWindow': 10,          # Short MA period
    'longWindow': 20,           # Long MA period
    'RSIWindow': 14,            # RSI period
    'bollingerBands': 20,       # Bollinger Bands period
    'ATR': 14,                  # ATR period
    'stochasticOscillator': 14  # Stochastic period
}
```

## Known Limitations

1. **Transaction Costs**: Not included in calculations
2. **Slippage**: Not modeled
3. **Market Hours**: Uses daily data only
4. **Survivorship Bias**: Only tests on existing stocks
5. **Overfitting**: Parameters not optimized for specific stocks

---

**Last Updated**: December 2024
**Version**: 2.0 (Fixed and Documented)
