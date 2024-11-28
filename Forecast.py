from statsmodels.tsa.stattools import adfuller
import pandas as pd
import warnings
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA

def isStationarity(funcData):
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(funcData)
    print("Computing the Augmented Dickey-Fuller statistical test...")
    print(f"Test statistic is {result[1]}")
    print("")
    return result[1]

def chooseARIMAParameter(funcData, targetColumn, max_p=6, max_d=4, max_q=6):
    warnings.filterwarnings("ignore")
    
    if isinstance(funcData, pd.DataFrame):
        if targetColumn not in funcData.columns:
            raise ValueError(f"Column '{targetColumn}' not found in the dataset.")
        tempData = funcData[targetColumn]
    else:
        tempData = funcData
    
    # Initial stationarity check
    p_value = isStationarity(tempData)
    
    differencing_steps = 0
    
    while p_value > 0.10 and differencing_steps < max_d:
        print(f"p-value = {p_value:.5f}. The series is non-stationary. Applying differencing...")
        tempData = tempData.diff().dropna()  # First-order differencing
        differencing_steps += 1
        p_value = isStationarity(tempData)
        print(f"After differencing {differencing_steps} time(s), p-value is now {p_value:.5f}.")

    print("")
    
    if p_value <= 0.10:
        print("The series is stationary. Proceeding with modeling.")
    else:
        print(f"The series is still non-stationary after {max_d} differencing steps. Consider alternative transformations.")

    # Define training and testing data
    train_size = int(len(tempData) * 0.8)  # Typically 80% for training
    train, test = tempData[:train_size], tempData[train_size:]
    
    # Initialize AIC score tracking for model selection
    best_aic = float('inf')
    best_order = None
    
    print("This may take a moment.")
    print(f"Training model for {targetColumn}...")
    print("")
    
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
    
    if best_order is None:
        print("No valid ARIMA model found. Returning default order (1, 0, 0).")
        print("")
        return (1, 0, 0), None
    else:
        print(f"Best ARIMA model order: {best_order} with AIC: {best_aic}")
        print("")
        # print(best_model)
        warnings.resetwarnings()        
        return best_order

def forecastPrices(funcData, num_days):
    print("Generating forecast...")
    print("")
    
    workingData = funcData
    
    if not isinstance(workingData.index, pd.DatetimeIndex):
        if 'Date' in workingData.columns:
            workingData['Date'] = pd.to_datetime(workingData['Date'])
            workingData.set_index('Date', inplace=True)
        else:
            raise ValueError("Data must have a 'Date' column or a datetime index.")
    workingData = workingData.asfreq('D').interpolate(method='time')
    
    priceOrder = chooseARIMAParameter(workingData, 'Close')
    volumeOrder = chooseARIMAParameter(workingData, 'Volume')
    
    warnings.filterwarnings("ignore")
    
    # Price forecast using ARIMA
    price_model = ARIMA(workingData['Close'], order=priceOrder)
    price_model_fit = price_model.fit()
    price_forecast = price_model_fit.forecast(steps=num_days)
    
    # Volume forecast using ARIMA
    volume_model = ARIMA(workingData['Volume'], order=volumeOrder)
    volume_model_fit = volume_model.fit()
    volume_forecast = volume_model_fit.forecast(steps=num_days)
    
    warnings.resetwarnings()
    
    # Create a date range for forecasting
    last_date = workingData.index[-1]
    trading_dates = pd.date_range(start=last_date + BDay(1), periods=num_days, freq='B')

    forecast = []
    for i in range(num_days):
        forecasted_price = price_forecast[i]
        forecasted_volume = volume_forecast[i]
        
        # Simulate High and Low prices based on Close price and a percentage range
        # Here, we're assuming that the High is 2% above Close, and Low is 2% below Close.
        forecasted_high = forecasted_price * 1.02
        forecasted_low = forecasted_price * 0.98
        
        forecast_date = trading_dates[i]
        forecast.append([forecast_date, forecasted_price, forecasted_high, forecasted_low, forecasted_volume])

    # Create the DataFrame with forecasted data
    forecast_df = pd.DataFrame(forecast, columns=['Date', 'Close', 'High', 'Low', 'Volume'])
    forecast_df.set_index('Date', inplace=True)

    print("Forecast generated. \n")
    return forecast_df