import pandas as pd
import warnings
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import os
from statsmodels.tsa.stattools import adfuller
from dateutil.relativedelta import relativedelta

def workWithData(symbol):
    print(f"Options for analysis of {symbol}:")
    print(" (1) Delete save data,")
    print(" (2) View save data.")
    print(" (3) Delete save data,")
    
def isStationarity(data):
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(data)
    print("Computing the Augmented Dickey-Fuller statistical test...")
    print(f"Test statistic is {result[1]}")
    print("")
    return result[1]

def chooseARIMAParameter(data, target_column='Close', max_p=6, max_d=2, max_q=6):
    choice = input("When training the ARIMA model to produce our forecast, many warnings are likely to occur. Would you like to view warnings or ignore them (I - ignore, anything else - dont ignore)? ")
    if choice == "i" or choice == "I":
        warnings.filterwarnings("ignore")
    
    print("")
    print("This may take a moment.")
    print("Training model...")
    print("")
            
    # Ensure the target_column exists
    if isinstance(data, pd.DataFrame):
        if target_column not in data.columns:
            raise ValueError(f"Column '{target_column}' not found in the dataset.")
        tempData = data[target_column]
    else:
        tempData = data
    
    # Initial stationarity check
    p_value = isStationarity(tempData)
    
    differencing_steps = 0
    
    # Perform differencing while the data is non-stationary and we haven't exceeded the max differencing
    while p_value > 0.05 and differencing_steps < max_d:
        tempData = tempData.diff().dropna()  # First-order differencing
        differencing_steps += 1
        p_value = isStationarity(tempData)
        print(f"After differencing {differencing_steps} times, p-value is: {p_value}")

    # If the series is still non-stationary after max differencing attempts, print a warning
    print("")
    if p_value > 0.05:
        print(f"Data remains non-stationary after {max_d} differencing steps. Consider other transformations.")
    else:
        print(f"Data became stationary after {differencing_steps} differencing steps.")

    # Define training and testing data
    train_size = int(len(tempData) * 0.8)  # Typically 80% for training
    train, test = tempData[:train_size], tempData[train_size:]
    
    # Initialize AIC score tracking for model selection
    best_aic = float('inf')
    best_order = None
    
    # Hyperparameter grid search over p, d, q
    for p in range(1, max_p + 1):
        for d in range(differencing_steps + 1):  # Based on the differencing steps found
            for q in range(1, max_q + 1):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    print(f"Error fitting model ({p}, {d}, {q}): {e}")
    
    print("")
    if best_order is None:
        print("No valid ARIMA model found. Returning default order (1, 0, 0).")
        return (1, 0, 0), None
    else:
        print(f"Best ARIMA model order: {best_order} with AIC: {best_aic}")
        # print(best_model)
        warnings.resetwarnings()        
        return best_order

def forecastPrices(data, num_days=365):
    print("")
    print("")
    print("Generating forecast...")
    print("")
    print("")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        else:
            raise ValueError("Data must have a 'Date' column or a datetime index.")
    data = data.asfreq('D').interpolate(method='time')
    
    Order = chooseARIMAParameter(data)
    
    warnings.filterwarnings("ignore")
    
    price_model = ARIMA(data['Close'], order=Order)
    price_model_fit = price_model.fit()
    price_forecast = price_model_fit.forecast(steps=num_days)
    volume_model = ARIMA(data['Volume'], order=Order)
    volume_model_fit = volume_model.fit()
    volume_forecast = volume_model_fit.forecast(steps=num_days)
    
    warnings.resetwarnings()

    last_date = data.index[-1]
    trading_dates = pd.date_range(start=last_date + BDay(1), periods=num_days, freq='B')
    
    # Collect forecasts
    forecast = []
    for i in range(num_days):
        forecasted_price = price_forecast.iloc[i] if isinstance(price_forecast, pd.Series) else price_forecast[i]
        forecasted_volume = volume_forecast.iloc[i] if isinstance(volume_forecast, pd.Series) else volume_forecast[i]
        forecast_date = trading_dates[i]
        forecast.append([forecast_date, forecasted_price, forecasted_volume])

    # Create DataFrame
    forecast_df = pd.DataFrame(forecast, columns=['Date', 'Close', 'Volume'])
    forecast_df.set_index('Date', inplace=True)

    print("")
    print("Forecast generated.")
    print("")
    print(forecast_df)
    return forecast_df

def getSymbol():
    while True:
        symbol = input("Enter symbol: ").strip().upper()      
        try:
            data = yf.Ticker(symbol).history(period="1d")
            if data.empty:
                print(f"Invalid symbol: {symbol}. Please try again.")
            else:
                print(f"Successfully fetched data for {symbol}.")
                return symbol
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")


def getData(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

def isEmpty(data):
    if data is None:
        print("Failed to fetch data. Check ticker or network connection.")
        exit()
        
def backTest(data, money, method):
    print("")
    print("")
    print("Back testing data...")
    close = data['Close'].values.tolist()
    close = [x[0] for x in close]
    percentReturns = []
    
    for i in range(1, len(close)):
        percentReturns.append(close[i]/close[i-1])
        
    percentReturns.insert(0, 1)    
    data['%Diff'] = percentReturns
    
    cumulativePercentageReturns = [1]
    for i in range(1, len(percentReturns)):
        cumulativePercentageReturns.append(cumulativePercentageReturns[-1] * percentReturns[i])
        
    data['%TotDiff'] = cumulativePercentageReturns
    
    portfolioValue = [money]
    
    for i in range(1, len(percentReturns)):
        portfolioValue.append(portfolioValue[-1] * percentReturns[i])
        
    data['Portfolio'] = portfolioValue
    
    if method is None:
        print("No method has been specified")
        print("")
        print("")
    else:
        print(f"{method} is being used")

    return data

def computeMovingAverages(data, shortWindow, longWindow):
    dataWithMA = data
    close = data['Close'].values.tolist()
    
    sMA = [0]
    lMA = [0]
    
    for endPoint in range(1, len(close)):
        if endPoint < shortWindow:
            temp = np.array(close[:endPoint])
            total_sum = temp.sum()
            MA = total_sum / endPoint
            sMA.append(MA)
        else:
            temp = np.array(close[endPoint - shortWindow:endPoint])
            total_sum = temp.sum()
            MA = total_sum / shortWindow
            sMA.append(MA)
        
    #generate lMA list
    for endPoint in range(1, len(close)):
        if endPoint < longWindow:
            temp = np.array(close[:endPoint])
            total_sum = temp.sum()
            MA = total_sum / endPoint
            lMA.append(MA)
        else:
            temp = np.array(close[endPoint - longWindow:endPoint])
            total_sum = temp.sum()
            MA = total_sum / longWindow
            lMA.append(MA)
            
    dataWithMA['MAS'] = sMA
    dataWithMA['MAL'] = lMA    

    signals = [0]
    sMA = dataWithMA['MAS'].values.tolist()
    lMA = dataWithMA['MAL'].values.tolist()
    weight = 1
    
    for i in range(1, len(sMA)):
        if sMA[i] > lMA[i]:
            signals.append(1)
        elif sMA[i] < lMA[i]:
            signals.append(-1)
        else:
            signals.append(0)
            
    dataWithMA['SigMA'] = signals
    
    return dataWithMA
    

def generateStrategy(data, shortWindow, longWindow):
    # dataWithMA = computeMovingAverages(data, shortWindow, longWindow)
            
    return computeMovingAverages(data, shortWindow, longWindow)

def predictions(data, shortWindow, longWindow):
    print("")
    print("")
    print("Making Predictions...")
    print("")
    print("")
    
    forecastedPrices = forecastPrices(data, num_days=365)
    predictions = []
     
    predictionsDataTested = generateStrategy(forecastedPrices, shortWindow, longWindow)
    predictionsDataTested.reset_index(inplace=True)

    predictionClose = predictionsDataTested['Close'].values.tolist()
    predictionSignal = predictionsDataTested['SigMA'].values.tolist()
    predictionDate = predictionsDataTested['Date'].values.tolist()
    
    for close, signal, date in zip(predictionClose, predictionSignal, predictionDate):
        predictions.append([date, close, signal])
    
    print("")
    print("")
    print("Predictions Made.")
    print("")
    print("")
    return predictions, predictionsDataTested

def setParameters():
    return 20, 40

def outputToFiles(data, symbol, end_date, shortWindow, longWindow):
    
    predict, analysisData = predictions(data, shortWindow, longWindow)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(current_dir, f"Analysis_of_{symbol}")
    os.makedirs(output_dir, exist_ok=True)

    predictionsOutputFile = os.path.join(output_dir, f"trade_predictions_{symbol}_{end_date}.txt")
    analysisOutputFile = os.path.join(output_dir, f"analysis_used_{symbol}_{end_date}.txt")
    dataOutputFile = os.path.join(output_dir, f"data_used_{symbol}_{end_date}.txt")

    with open(predictionsOutputFile, "w") as f:
        for item in predict:
            f.write(','.join(map(str, item)) + '\n')
    
    with open(analysisOutputFile, "w") as f:
        f.write(analysisData.to_string())
        
    with open(dataOutputFile, "w") as f:
        f.write(data.to_string())
        
    return predictionsOutputFile, dataOutputFile

def getDates(start, end):
    if end == 0:
        end_date = datetime.today().strftime('%Y-%m-%d')
    else:
        print("Additional time commands soon.")
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    start_date = (datetime.today() - relativedelta(years=start)).strftime('%Y-%m-%d')

    return start_date, end_date

def getPercentages(data, money):
    close = data['Close'].values.tolist()
    close = [x[0] for x in close]
    percentReturns = []
    
    for i in range(1, len(close)):
        percentReturns.append(close[i]/close[i-1])
        
    percentReturns.insert(0, 1)    
    data['%Diff'] = percentReturns
    
    cumulativePercentageReturns = [1]
    for i in range(1, len(percentReturns)):
        cumulativePercentageReturns.append(cumulativePercentageReturns[-1] * percentReturns[i])
        
    data['%TotDiff'] = cumulativePercentageReturns
    
    portfolioValue = [money]
    
    for i in range(1, len(percentReturns)):
        portfolioValue.append(portfolioValue[-1] * percentReturns[i])
        
    data['Portfolio'] = portfolioValue

    return data

def main():
    symbol = getSymbol()
    
    '''
    0 returns today, a number gives the number of years ago to start the data from
    '''
    start = 5
    while True:
        try:
            start = int(input("When would you like to test from (1 - 1 year ago, 2 - 2 years ago, ...)? "))
        except:
            print("Input not an integer, choose again")
        if start < 1:
            print("Cannot accept negatives")
        else:
            break
    
    start_date, end_date = getDates(start, 0)
    
    data = getData(symbol, start_date, end_date)
    
    isEmpty(data)
    
    print("")
    print("")
    print("Generating strategy...")
    print("")
    print("")
    
    shortWindow, longWindow = setParameters()
        
    testData = generateStrategy(data, shortWindow, longWindow)
    
    startSum = 100000
    
    historicalTotalReturns = getPercentages(data, startSum)
    strategyTotalReturns = backTest(testData, startSum, None)
    
    predictionsOutputFile, dataOutputFile = outputToFiles(data, symbol, end_date, shortWindow, longWindow)
    
    
    print("")
    print("")
    while True:
        choice0 = input("See Historical and Strategy data (y/n)? ")
        print("")
        print("")
        if choice0 == "y":
            print("Historical Returns")        
            print(historicalTotalReturns)
            print("")
            print("")
            print("Strategic Returns")
            print(strategyTotalReturns)
            break
        elif choice0 == "n":
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            
    print("")
    print("")
    print(f"Predictions saved to trade_predictions_{symbol}_{end_date}.")
    print(f"Analysis used saved to analysis_used_{symbol}_{end_date}.")
    print(f"Data used saved to data_used_{symbol}_{end_date}.")
    print("")
    print("")
    print(f"Historically, returns from {start_date} to {end_date} is {historicalTotalReturns['%TotDiff'].iloc[-1]}")
    print(f"Using the strategy, returns from {start_date} to {end_date} is {strategyTotalReturns['%TotDiff'].iloc[-1]}")
    print("")
    print("")
    
    x = True
    while x:
        choice = input("Exit (1), work with data (2) or try again (3): ")
        if choice == "1":
            x = False
            break
        elif choice == "2":
            workWithData(symbol)
        elif choice == "3":
            main()
            x = False
        else:
            continue

if __name__ == "__main__":
    print("""
    ╭━━━━╮      ╭╮ ╭━━━━╮     ╭╮
    ┃╭╮╭╮┃      ┃┃ ┃╭╮╭╮┃     ┃┃
    ╰╯┃┃┣┻━┳━━┳━╯┃ ╰╯┃┃┣┻┳━━┳━╯┣━━┳━╮
      ┃┃┃╭╮┃╭╮┃╭╮┃   ┃┃┃╭┫╭╮┃╭╮┃┃━┫╭╯
      ┃┃┃╰╯┃╭╮┃╰╯┃   ┃┃┃┃┃╭╮┃╰╯┃┃━┫┃
      ╰╯╰━━┻╯╰┻━━╯   ╰╯╰╯╰╯╰┻━━┻━━┻╯
    """)
    main()
