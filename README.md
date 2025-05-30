# TOADTrade - Technical Analysis Trading Algorithm

A comprehensive Python-based trading algorithm that uses multiple technical indicators to generate trading signals and backtest strategies on historical stock data.

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

✅ **Real-time Data Fetching**: Uses yfinance to fetch current stock data
✅ **Multiple Technical Indicators**: 7 different indicators with proper signal generation
✅ **Weighted Signal System**: Combines all indicators for comprehensive analysis
✅ **Backtesting Engine**: Tests strategy performance against buy-and-hold
✅ **Performance Metrics**: Clear comparison of strategy vs market returns
✅ **Forecasting**: SARIMAX-based price prediction (experimental)
✅ **Comprehensive Documentation**: Inline comments and clear function descriptions

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install yfinance pandas numpy statsmodels python-dateutil
```

## Usage

### Basic Usage
Run the main program:
```bash
python Main.py
```

The program will:
1. Ask for a stock symbol (e.g., AAPL, TSLA, MSFT)
2. Ask for the analysis period (1-5 years)
3. Fetch historical data
4. Generate technical indicators and signals
5. Run backtesting simulation
6. Display performance comparison
7. Generate forecasts
8. Save results to files

### Example Output
```
Trading Strategy Results:
  Initial capital: $1,000.00
  Final portfolio value: $1,023.33
  Total return: 2.33%
  Total trades: 7 (Buy: 4, Sell: 3)
  Final position: $0.00 cash + 4.31 shares

Performance calculation complete:
  Initial investment: $1,000.00
  Final portfolio value: $1,064.69
  Total return: 6.47%
```

## File Structure

- **Main.py**: Core application with data fetching, signal generation, and backtesting
- **Indicators.py**: Technical indicator calculations and signal generation
- **Forecast.py**: SARIMAX-based price forecasting functionality
- **README.md**: This documentation file

## Technical Indicators Explained

### 1. Moving Averages (MA)
- **Short MA > Long MA**: Buy signal (uptrend)
- **Short MA < Long MA**: Sell signal (downtrend)
- Default: 10-period vs 20-period

### 2. Relative Strength Index (RSI)
- **RSI < 30**: Buy signal (oversold)
- **RSI > 70**: Sell signal (overbought)
- Default: 14-period

### 3. MACD
- **MACD > Signal Line**: Buy signal
- **MACD < Signal Line**: Sell signal
- Default: 12/26/9 EMA configuration

### 4. Bollinger Bands
- **Price < Lower Band**: Buy signal (oversold)
- **Price > Upper Band**: Sell signal (overbought)
- Default: 20-period with 2 standard deviations

### 5. On-Balance Volume (OBV)
- **Price up + Volume**: Buy signal
- **Price down + Volume**: Sell signal

### 6. Average True Range (ATR)
- **Increasing ATR**: Buy signal (increasing volatility)
- **Decreasing ATR**: Sell signal (decreasing volatility)
- Default: 14-period

### 7. Stochastic Oscillator
- **%K crosses above %D**: Buy signal
- **%K crosses below %D**: Sell signal
- Default: 14-period with 3-period %D

## Trading Strategy

The algorithm uses a **weighted signal approach**:

1. **Signal Generation**: Each indicator generates signals (-1, 0, +1)
2. **Signal Combination**: All signals are summed to create a weighted signal
3. **Trading Decisions**:
   - **Signal ≥ threshold**: Buy (invest available cash)
   - **Signal ≤ -threshold**: Sell (liquidate position)
   - **Otherwise**: Hold current position

### Risk Management
- **Tolerance Levels**: 1 (conservative) to 5 (very risky)
- **Position Sizing**: All-in/all-out approach (can be modified)
- **Transaction Costs**: Not currently included (can be added)

## Performance Metrics

The system calculates and compares:
- **Buy-and-Hold Returns**: Simple market performance
- **Strategy Returns**: Algorithm-based trading performance
- **Total Trades**: Number of buy/sell transactions
- **Final Position**: Cash and shares held

## Customization

### Modify Indicator Parameters
Edit the `parameters` dictionary in `main()`:
```python
parameters = {
    'shortWindow': 10,      # Short MA period
    'longWindow': 20,       # Long MA period
    'RSIWindow': 14,        # RSI period
    'bollingerBands': 20,   # Bollinger Bands period
    'ATR': 14,              # ATR period
    'stochasticOscillator': 14  # Stochastic period
}
```

### Adjust Risk Tolerance
Modify the `getTolerance()` function or make it interactive.

### Add New Indicators
1. Create new function in `Indicators.py`
2. Add to the indicator chain in `method1()`
3. Ensure it returns a signal column starting with 'Sig'

## Known Limitations

1. **Transaction Costs**: Not included in calculations
2. **Slippage**: Not modeled
3. **Market Hours**: Uses daily data only
4. **Survivorship Bias**: Only tests on existing stocks
5. **Overfitting**: Parameters not optimized for specific stocks

## Future Improvements

- [ ] Add transaction cost modeling
- [ ] Implement position sizing strategies
- [ ] Add more technical indicators
- [ ] Create parameter optimization
- [ ] Add real-time trading capabilities
- [ ] Implement risk management rules
- [ ] Add portfolio diversification
- [ ] Create web interface

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading stocks involves risk, and you may lose money. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## License

This project is open source. Feel free to modify and distribute according to your needs.

---

**Last Updated**: December 2024
**Version**: 2.0 (Fixed and Documented)
