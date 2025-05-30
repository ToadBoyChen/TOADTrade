import numpy as np
import pandas as pd

def computeMovingAverages(funcData, shortWindow, longWindow):
    """
    Compute moving averages and generate trading signals.
    
    Args:
        funcData (pandas.DataFrame): Stock data with 'Close' column
        shortWindow (int): Period for short-term moving average
        longWindow (int): Period for long-term moving average
    
    Returns:
        pandas.DataFrame: Data with moving average signals added
    """
    dataWithMA = funcData.copy()
    
    # Calculate moving averages using pandas rolling function (more efficient)
    dataWithMA['MAS'] = dataWithMA['Close'].rolling(window=shortWindow, min_periods=1).mean()
    dataWithMA['MAL'] = dataWithMA['Close'].rolling(window=longWindow, min_periods=1).mean()
    
    # Generate signals: 1 for buy (short MA > long MA), -1 for sell (short MA < long MA), 0 for hold
    signals = []
    for i in range(len(dataWithMA)):
        short_ma = dataWithMA['MAS'].iloc[i]
        long_ma = dataWithMA['MAL'].iloc[i]
        
        if short_ma > long_ma:
            signals.append(1)  # Buy signal
        elif short_ma < long_ma:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # Hold signal
    
    dataWithMA['SigMA'] = signals
    
    # Clean up temporary columns
    dataWithMA = dataWithMA.drop(columns=['MAS', 'MAL'])
    
    print(f"Moving Average signals: Short={shortWindow}, Long={longWindow} periods")
    return dataWithMA

def computeRSI(data, window):
    """
    Compute Relative Strength Index (RSI) and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' column
        window (int): Period for RSI calculation
    
    Returns:
        pandas.DataFrame: Data with RSI signals added
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi.fillna(0)

    signals = [0] * len(data)

    # Generate signals: RSI < 30 = oversold (buy), RSI > 70 = overbought (sell)
    for i in range(len(data['RSI'])):
        if data['RSI'].iloc[i] < 30:
            signals[i] = 1  # Buy signal (oversold)
        elif data['RSI'].iloc[i] > 70:
            signals[i] = -1  # Sell signal (overbought)
    
    data['SigRSI'] = signals
    
    # Clean up temporary column
    data = data.drop(columns=['RSI'])
    
    print(f"RSI signals: {window} period window (Buy<30, Sell>70)")
    return data

def computeMACD(data):
    """
    Compute MACD (Moving Average Convergence Divergence) and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' column
    
    Returns:
        pandas.DataFrame: Data with MACD signals added
    """
    # Calculate MACD components
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()   # 26-period EMA
    macd = shortEMA - longEMA                                  # MACD line
    signal = macd.ewm(span=9, adjust=False).mean()             # Signal line (9-period EMA of MACD)
    
    data['MACD'] = macd
    data['SigLine'] = signal

    signals = []
    # Generate signals: MACD above signal line = buy, below = sell
    for i in range(len(data['MACD'])):
        if data['MACD'].iloc[i] > data['SigLine'].iloc[i]:
            signals.append(1)   # Buy signal
        elif data['MACD'].iloc[i] < data['SigLine'].iloc[i]:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)   # Hold signal

    data['SigMACD'] = signals
    
    # Clean up temporary columns
    data = data.drop(columns=['MACD', 'SigLine'])
    
    print(f"MACD signals: 12/26/9 EMA crossover strategy")
    return data

def computeBollingerBands(data, window):
    """
    Compute Bollinger Bands and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' column
        window (int): Period for moving average and standard deviation
    
    Returns:
        pandas.DataFrame: Data with Bollinger Band signals added
    """
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    data['UpBand'] = sma + (2 * std)
    data['LoBand'] = sma - (2 * std)
    data['SMA_BB'] = sma
    data[['UpBand', 'LoBand', 'SMA_BB']] = data[['UpBand', 'LoBand', 'SMA_BB']].fillna(0)
    
    signals = []
    
    # Generate signals based on price relative to bands
    for i in range(len(data)):
        close_price = data['Close'].iloc[i]
        upper_band = data['UpBand'].iloc[i]
        lower_band = data['LoBand'].iloc[i]
        
        if close_price < lower_band:
            signals.append(1)  # Buy signal (oversold)
        elif close_price > upper_band:
            signals.append(-1)  # Sell signal (overbought)
        else:
            signals.append(0)  # Hold signal
                
    data['SigBB'] = signals
    
    # Clean up temporary columns
    data = data.drop(columns=['UpBand', 'LoBand', 'SMA_BB'])
    
    print(f"Bollinger Bands signals: {window} period window")
    return data


def computeOBV(data):
    """
    Compute On-Balance Volume (OBV) and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'Close' and 'Volume' columns
    
    Returns:
        pandas.DataFrame: Data with OBV signals added
    """
    obv = [0]
    signals = [0]

    # Calculate OBV and signals based on price direction
    for i in range(1, len(data)):
        current_close = data['Close'].iloc[i]
        previous_close = data['Close'].iloc[i - 1]
        current_volume = data['Volume'].iloc[i]
        
        if current_close > previous_close:
            # Price up: add volume to OBV, buy signal
            obv.append(obv[-1] + current_volume)
            signals.append(1)
        elif current_close < previous_close:
            # Price down: subtract volume from OBV, sell signal
            obv.append(obv[-1] - current_volume)
            signals.append(-1)
        else:
            # Price unchanged: OBV unchanged, hold signal
            obv.append(obv[-1])
            signals.append(0)

    data['SigOBV'] = signals
    
    print(f"OBV signals: Volume-based momentum indicator")
    return data

def computeATR(data, window):
    """
    Compute Average True Range (ATR) and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'High', 'Low', 'Close' columns
        window (int): Period for ATR calculation
    
    Returns:
        pandas.DataFrame: Data with ATR signals added
    """
    # Calculate True Range components
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    
    # True Range is the maximum of the three components
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    
    # ATR is the moving average of True Range
    atr = true_range.rolling(window=window).mean()
    data['ATR'] = atr

    signals = []
    # Generate signals: increasing ATR = volatility increasing (buy), decreasing = sell
    for i in range(1, len(atr)):
        if data['ATR'].iloc[i] > data['ATR'].iloc[i-1]:
            signals.append(1)   # Buy signal (increasing volatility)
        else:
            signals.append(-1)  # Sell signal (decreasing volatility)
    signals.insert(0, 0)  # First day has no signal

    data['SigATR'] = signals
    
    # Clean up temporary column
    data = data.drop(columns=['ATR'])
    
    print(f"ATR signals: {window} period volatility indicator")
    return data

def computeStochasticOscillator(data, window):
    """
    Compute Stochastic Oscillator and generate trading signals.
    
    Args:
        data (pandas.DataFrame): Stock data with 'High', 'Low', 'Close' columns
        window (int): Period for stochastic calculation
    
    Returns:
        pandas.DataFrame: Data with Stochastic signals added
    """
    # Calculate %K and %D lines
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()  # 3-period moving average of %K

    signals = []
    # Generate signals based on %K and %D crossovers
    for i in range(1, len(data)):
        k_current = data['%K'].iloc[i]
        d_current = data['%D'].iloc[i]
        k_previous = data['%K'].iloc[i - 1]
        d_previous = data['%D'].iloc[i - 1]
        
        if k_current > d_current and k_previous <= d_previous:
            signals.append(1)   # Buy signal (%K crosses above %D)
        elif k_current < d_current and k_previous >= d_previous:
            signals.append(-1)  # Sell signal (%K crosses below %D)
        else:
            signals.append(0)   # Hold signal
    
    signals.insert(0, 0)  # First day has no signal

    data['SigSto'] = signals
    
    # Clean up temporary columns
    data = data.drop(columns=['%K', '%D'])
    
    print(f"Stochastic Oscillator signals: {window} period window with 3-period %D")
    return data
