from statsmodels.tsa.stattools import adfuller
import pandas as pd
import warnings
from pandas.tseries.offsets import BDay
from statsmodels.tsa.statespace.sarimax import SARIMAX

def isStationarity(funcData):
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(funcData)
    print("Computing the Augmented Dickey-Fuller statistical test...")
    print(f"Test statistic is {result[1]}")
    print("")
    return result[1]

def chooseSARIMAXParameter(funcData, targetColumn, max_p, max_d, max_q, max_P, max_D, max_Q, m):
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
    train_size = int(len(tempData) * 0.8)
    train, test = tempData[:train_size], tempData[train_size:]

    # Initialize AIC score tracking
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None

    print("Searching for the best SARIMAX parameters...")
    
    # Hyperparameter grid search over p, d, q, P, D, Q
    for p in range(max_p + 1):
        for d in range(differencing_steps + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(
                                    train,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, m),
                                )
                                model_fit = model.fit(disp=False)
                                aic = model_fit.aic
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, m)
                                    best_model = model_fit
                            except Exception as e:
                                pass

    if best_order is None:
        print("No valid SARIMAX model found. Returning default order (1, 0, 0).")
        return (1, 0, 0), (0, 0, 0, 0), None
    else:
        print(f"Best SARIMAX model order: {best_order}, Seasonal order: {best_seasonal_order} with AIC: {best_aic}")
        return best_order, best_seasonal_order, best_model

def forecastPricesHighLowVolume(funcData, num_days, m):
    print("Generating forecast for 'High', 'Low', 'Close', and 'Volume'...")
    print("")
    
    if not isinstance(funcData.index, pd.DatetimeIndex):
        if 'Date' in funcData.columns:
            funcData['Date'] = pd.to_datetime(funcData['Date'])
            funcData.set_index('Date', inplace=True)
        else:
            raise ValueError("Data must have a 'Date' column or a datetime index.")
    funcData = funcData.asfreq('D').interpolate(method='time')
    
    max_p, max_d, max_q = 2, 1, 2
    max_P, max_D, max_Q = 1, 1, 1 

    # SARIMAX for 'Close'
    closeOrder, closeSeasonalOrder, _ = chooseSARIMAXParameter(funcData, 'Close', max_p, max_d, max_q, max_P, max_D, max_Q, m)
    
    # SARIMAX for 'Volume'
    volumeOrder, volumeSeasonalOrder, _ = chooseSARIMAXParameter(funcData, 'Volume', max_p, max_d, max_q, max_P, max_D, max_Q, m)

    # SARIMAX for 'High'
    highOrder, highSeasonalOrder, _ = chooseSARIMAXParameter(funcData, 'High', max_p, max_d, max_q, max_P, max_D, max_Q, m)

    # SARIMAX for 'Low'
    lowOrder, lowSeasonalOrder, _ = chooseSARIMAXParameter(funcData, 'Low', max_p, max_d, max_q, max_P, max_D, max_Q, m)

    warnings.filterwarnings("ignore")
    
    # Forecast with SARIMAX for 'Close'
    close_model = SARIMAX(funcData['Close'], order=closeOrder, seasonal_order=closeSeasonalOrder)
    close_model_fit = close_model.fit()
    close_forecast = close_model_fit.forecast(steps=num_days)
    
    # Forecast with SARIMAX for 'Volume'
    volume_model = SARIMAX(funcData['Volume'], order=volumeOrder, seasonal_order=volumeSeasonalOrder)
    volume_model_fit = volume_model.fit()
    volume_forecast = volume_model_fit.forecast(steps=num_days)

    # Forecast with SARIMAX for 'High'
    high_model = SARIMAX(funcData['High'], order=highOrder, seasonal_order=highSeasonalOrder)
    high_model_fit = high_model.fit()
    high_forecast = high_model_fit.forecast(steps=num_days)

    # Forecast with SARIMAX for 'Low'
    low_model = SARIMAX(funcData['Low'], order=lowOrder, seasonal_order=lowSeasonalOrder)
    low_model_fit = low_model.fit()
    low_forecast = low_model_fit.forecast(steps=num_days)
    
    warnings.resetwarnings()

    # Create a date range for forecasting
    last_date = funcData.index[-1]
    forecast_dates = pd.date_range(start=last_date + BDay(1), periods=num_days, freq='B')

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'High': high_forecast,
        'Low': low_forecast,
        'Close': close_forecast,
        'Volume': volume_forecast
    }).set_index('Date')

    print("Forecast generated.")
    return forecast_df
