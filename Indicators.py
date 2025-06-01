import numpy as np
import pandas as pd

def computeMovingAverages(funcData, shortWindow, longWindow):
    """
    Compute moving averages and generate trading signals.
    """
    if funcData.empty or 'Close' not in funcData.columns:
        print("Insufficient data for Moving Averages computation.")
        return funcData
    
    dataWithMA = funcData.copy()
    dataWithMA['MAS'] = dataWithMA['Close'].rolling(window=shortWindow, min_periods=1).mean()
    dataWithMA['MAL'] = dataWithMA['Close'].rolling(window=longWindow, min_periods=1).mean()
    
    signals = []
    for i in range(len(dataWithMA)):
        short_ma = dataWithMA['MAS'].iloc[i]
        long_ma = dataWithMA['MAL'].iloc[i]
        if short_ma > long_ma:
            signals.append(1)
        elif short_ma < long_ma:
            signals.append(-1)
        else:
            signals.append(0)
    
    dataWithMA['SigMA'] = signals
    dataWithMA = dataWithMA.drop(columns=['MAS', 'MAL'])
    return dataWithMA

def computeRSI(data, window):
    """
    Compute Relative Strength Index (RSI) and generate trading signals.
    """
    if data.empty or 'Close' not in data.columns:
        print("Insufficient data for RSI computation.")
        return data
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi.fillna(0)
    signals = [0] * len(data)
    for i in range(len(data['RSI'])):
        if data['RSI'].iloc[i] < 30:
            signals[i] = 1
        elif data['RSI'].iloc[i] > 70:
            signals[i] = -1
    
    data['SigRSI'] = signals
    data = data.drop(columns=['RSI'])
    return data

def computeMACD(data):
    """
    Compute MACD (Moving Average Convergence Divergence) and generate trading signals.
    """
    if data.empty or 'Close' not in data.columns:
        print("Insufficient data for MACD computation.")
        return data
    
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    macd = shortEMA - longEMA
    signal = macd.ewm(span=9, adjust=False).mean()
    
    data['MACD'] = macd
    data['SigLine'] = signal

    signals = []
    for i in range(len(data['MACD'])):
        if data['MACD'].iloc[i] > data['SigLine'].iloc[i]:
            signals.append(1)
        elif data['MACD'].iloc[i] < data['SigLine'].iloc[i]:
            signals.append(-1)
        else:
            signals.append(0)

    data['SigMACD'] = signals
    data = data.drop(columns=['MACD', 'SigLine'])
    return data

def computeBollingerBands(data, window):
    """
    Compute Bollinger Bands and generate trading signals.
    """
    if data.empty or 'Close' not in data.columns:
        print("Insufficient data for Bollinger Bands computation.")
        return data
    
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    data['UpBand'] = sma + (2 * std)
    data['LoBand'] = sma - (2 * std)
    data['SMA_BB'] = sma
    data[['UpBand', 'LoBand', 'SMA_BB']] = data[['UpBand', 'LoBand', 'SMA_BB']].fillna(0)
    
    signals = []
    for i in range(len(data)):
        close_price = data['Close'].iloc[i]
        upper_band = data['UpBand'].iloc[i]
        lower_band = data['LoBand'].iloc[i]
        
        if close_price < lower_band:
            signals.append(1)
        elif close_price > upper_band:
            signals.append(-1)
        else:
            signals.append(0)
                
    data['SigBB'] = signals
    data = data.drop(columns=['UpBand', 'LoBand', 'SMA_BB'])
    return data

def computeOBV(data):
    """
    Compute On-Balance Volume (OBV) and generate trading signals.
    """
    if data.empty or 'Close' not in data.columns or 'Volume' not in data.columns:
        print("Insufficient data for OBV computation.")
        return data
    
    obv = [0]
    signals = [0]
    for i in range(1, len(data)):
        current_close = data['Close'].iloc[i]
        previous_close = data['Close'].iloc[i - 1]
        current_volume = data['Volume'].iloc[i]
        
        if current_close > previous_close:
            obv.append(obv[-1] + current_volume)
            signals.append(1)
        elif current_close < previous_close:
            obv.append(obv[-1] - current_volume)
            signals.append(-1)
        else:
            obv.append(obv[-1])
            signals.append(0)

    data['SigOBV'] = signals
    return data

def computeATR(data, window):
    """
    Compute Average True Range (ATR) and generate trading signals.
    """
    if data.empty or 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
        print("Insufficient data for ATR computation.")
        return data
    
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = true_range.rolling(window=window).mean()
    data['ATR'] = atr

    signals = []
    for i in range(1, len(atr)):
        if data['ATR'].iloc[i] > data['ATR'].iloc[i-1]:
            signals.append(1)
        else:
            signals.append(-1)
    signals.insert(0, 0)

    data['SigATR'] = signals
    data = data.drop(columns=['ATR'])
    return data

def computeStochasticOscillator(data, window):
    """
    Compute Stochastic Oscillator and generate trading signals.
    """
    if data.empty or 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
        print("Insufficient data for Stochastic Oscillator computation.")
        return data
    
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()

    signals = []
    for i in range(1, len(data)):
        k_current = data['%K'].iloc[i]
        d_current = data['%D'].iloc[i]
        k_previous = data['%K'].iloc[i - 1]
        d_previous = data['%D'].iloc[i - 1]
        
        if k_current > d_current and k_previous <= d_previous:
            signals.append(1)
        elif k_current < d_current and k_previous >= d_previous:
            signals.append(-1)
        else:
            signals.append(0)
    
    signals.insert(0, 0)

    data['SigSto'] = signals
    data = data.drop(columns=['%K', '%D'])
    return data
