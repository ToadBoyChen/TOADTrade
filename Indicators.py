import numpy as np
import pandas as pd

def computeMovingAverages(funcData, shortWindow, longWindow):
    dataWithMA = funcData.copy()
    close = funcData['Close'].values.tolist()
    
    sMA = [0]
    lMA = [0]
    
    for endPoint in range(1, len(close)):
        if endPoint < shortWindow:
            temp = np.array(close[:endPoint])
            MA = temp.sum() / endPoint
            sMA.append(MA)
        else:
            temp = np.array(close[endPoint - shortWindow:endPoint])
            MA = temp.sum() / shortWindow
            sMA.append(MA)
        
    for endPoint in range(1, len(close)):
        if endPoint < longWindow:
            temp = np.array(close[:endPoint])
            MA = temp.sum() / endPoint
            lMA.append(MA)
        else:
            temp = np.array(close[endPoint - longWindow:endPoint])
            MA = temp.sum() / longWindow
            lMA.append(MA)
            
    dataWithMA['MAS'] = sMA
    dataWithMA['MAL'] = lMA    

    signals = []
    for i in range(len(sMA)):
        if sMA[i] > lMA[i]:
            signals.append(1)
        elif sMA[i] < lMA[i]:
            signals.append(-1)
        else:
            signals.append(0)

    dataWithMA['SigMA'] = signals
    
    dataWithMA = dataWithMA.drop(columns=['MAS', 'MAL'])
    
    return dataWithMA

def computeRSI(data, window):
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
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    data['UpBand'] = sma + (2 * std)
    data['LoBand'] = sma - (2 * std)
    data['SMA_BB'] = sma
    data[['UpBand', 'LoBand', 'SMA_BB']] = data[['UpBand', 'LoBand', 'SMA_BB']].fillna(0)
    
    signals = []
    
    try:
        for i in range(len(data)):
            if data['Close'].iloc[i].values[0] < data['LoBand'].iloc[i]:
                signals.append(1)
            elif data['Close'].iloc[i].values[0] > data['UpBand'].iloc[i]:
                signals.append(-1)
            else:
                signals.append(0)
    except:
        for i in range(len(data)):
            if data['Close'].iloc[i] < data['LoBand'].iloc[i]:
                signals.append(1)
            elif data['Close'].iloc[i] > data['UpBand'].iloc[i]:
                signals.append(-1)
            else:
                signals.append(0)
                
    data['SigBB'] = signals
    
    data = data.drop(columns=['UpBand', 'LoBand', 'SMA_BB'])
    
    return data


def computeOBV(data):
    obv = [0]
    signals = [0]

    try:    
        for i in range(1, len(data)):
            if data['Close'].iloc[i].values[0] > data['Close'].iloc[i - 1].values[0]:
                obv.append(obv[-1] + data['Volume'].iloc[i].values[0])
                signals.append(1)
            elif data['Close'].iloc[i].values[0] < data['Close'].iloc[i - 1].values[0]:
                obv.append(obv[-1] - data['Volume'].iloc[i].values[0])
                signals.append(-1)
            else:
                obv.append(obv[-1])
                signals.append(0)
    except:
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
                signals.append(1)
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
                signals.append(-1)
            else:
                obv.append(obv[-1])
                signals.append(0)

    data['SigOBV'] = signals
    
    return data

def computeATR(data, window):
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
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()

    signals = []
    for i in range(1, len(data)):
        if data['%K'].iloc[i] > data['%D'].iloc[i] and data['%K'].iloc[i - 1] <= data['%D'].iloc[i - 1]:
            signals.append(1)
        elif data['%K'].iloc[i] < data['%D'].iloc[i] and data['%K'].iloc[i - 1] >= data['%D'].iloc[i - 1]:
            signals.append(-1)
        else:
            signals.append(0)
    signals.insert(0, 0)

    data['SigSto'] = signals
    
    data = data.drop(columns=['%K', '%D'])
    print(data)
    return data
