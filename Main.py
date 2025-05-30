from datetime import datetime
import yfinance as yf
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
import Indicators
import Forecast

def workWithData(symbol):
    print(f"Options for analysis of {symbol}:")
    print(" (1) Delete save data,")
    print(" (2) View save data.")
    print(" (3) Delete save data,")

def getSymbol():
    while True:
        symbol = input("Enter symbol: ").strip().upper()      
        try:
            funcData = yf.Ticker(symbol).history(period="1d")
            if funcData.empty:
                print(f"Invalid symbol: {symbol}. Please try again.")
            else:
                print(f"Successfully fetched data for {symbol}.")
                return symbol
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")

def getData(symbol, start_date, end_date):
    """
    Fetch historical stock data for a given symbol and date range.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pandas.DataFrame: Historical stock data with OHLCV columns
    """
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    # Download data with auto_adjust=False to get consistent column structure
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    # Handle multi-level columns from yf.download
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten multi-level columns by taking the first level (price type)
        data.columns = data.columns.get_level_values(0)
    
    isEmpty(data)
    print(f"Successfully fetched {len(data)} days of data.")
    return data

def isEmpty(funcData):
    if funcData is None:
        print("Failed to fetch data. Check ticker or network connection.")
        exit()
        
def method1(funcData, parameters):
    '''
    Method One so far uses:
        1. Moving averages 
        2. RSI
    '''
    movingAverages = Indicators.computeMovingAverages(funcData, parameters['shortWindow'], parameters['longWindow'])
    relStrengthIndex = Indicators.computeRSI(movingAverages, parameters['RSIWindow'])
    MACD = Indicators.computeMACD(relStrengthIndex)
    bollingerBands = Indicators.computeBollingerBands(MACD, parameters['bollingerBands'])
    OBV = Indicators.computeOBV(bollingerBands)
    ATR = Indicators.computeATR(OBV, parameters['ATR'])
    stochasticOscillator = Indicators.computeStochasticOscillator(ATR, parameters['stochasticOscillator'])
    return stochasticOscillator

def buyAndSell(funcData, money, weight):
    """
    Backtest a trading strategy based on weighted signals.
    
    Strategy Logic:
    - Signal >= weight: Buy signal (go long)
    - Signal <= -weight: Sell signal (go short or exit long)
    - Otherwise: Hold current position
    
    Args:
        funcData (pandas.DataFrame): Data with 'Signals' column
        money (float): Initial capital
        weight (float): Signal threshold for trading decisions
    
    Returns:
        pandas.DataFrame: Data with added trading simulation columns
    """
    workingData = funcData.copy()
    
    # Ensure we have percentage returns calculated
    if '%Diff' not in workingData.columns:
        workingData = getPercentages(workingData, money)
    
    signals = workingData['Signals'].values
    daily_returns = workingData['%Diff'].values  # Daily return factors (1.05 = 5% gain)
    
    # Initialize tracking variables
    portfolio_values = [money]  # Portfolio value over time
    cash_position = [money]     # Cash available
    stock_position = [0]        # Value of stock holdings
    shares_held = [0]           # Number of shares held
    total_trades = 0
    buy_trades = 0
    sell_trades = 0
    
    # Get close prices for share calculations
    close_prices = workingData['Close'].values
    
    for i in range(1, len(signals)):
        prev_cash = cash_position[-1]
        prev_shares = shares_held[-1]
        current_price = close_prices[i]
        
        # Calculate current value of stock holdings
        current_stock_value = prev_shares * current_price
        
        if signals[i] >= weight and prev_cash > 0:
            # Buy signal: invest available cash
            shares_to_buy = prev_cash / current_price
            new_shares = prev_shares + shares_to_buy
            new_cash = 0  # All cash invested
            buy_trades += 1
            total_trades += 1
            
        elif signals[i] <= -weight and prev_shares > 0:
            # Sell signal: sell all shares
            new_cash = prev_cash + (prev_shares * current_price)
            new_shares = 0
            sell_trades += 1
            total_trades += 1
            
        else:
            # Hold: maintain current position
            new_cash = prev_cash
            new_shares = prev_shares
        
        # Update tracking arrays
        cash_position.append(new_cash)
        shares_held.append(new_shares)
        stock_value = new_shares * current_price
        stock_position.append(stock_value)
        total_portfolio = new_cash + stock_value
        portfolio_values.append(total_portfolio)
    
    # Add results to dataframe
    workingData['Cash'] = cash_position
    workingData['Shares'] = shares_held
    workingData['StockValue'] = stock_position
    workingData['AlgoPort'] = portfolio_values
    
    # Calculate strategy returns
    strategy_returns = [portfolio_values[i] / portfolio_values[i-1] if i > 0 else 1 for i in range(len(portfolio_values))]
    workingData['StrategyReturns'] = strategy_returns
    
    # Calculate cumulative strategy performance
    cumulative_strategy_returns = pd.Series(strategy_returns).cumprod()
    workingData['%TotDiff'] = cumulative_strategy_returns
    
    final_value = portfolio_values[-1]
    total_return = (final_value / money - 1) * 100
    
    print(f"Trading Strategy Results:")
    print(f"  Initial capital: ${money:,.2f}")
    print(f"  Final portfolio value: ${final_value:,.2f}")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Total trades: {total_trades} (Buy: {buy_trades}, Sell: {sell_trades})")
    print(f"  Final position: ${cash_position[-1]:,.2f} cash + {shares_held[-1]:.2f} shares\n")
    
    return workingData


    
def backTest(funcData, money, tolerance):
    backTestData = funcData.copy()

    weight = tolerance/3
    
    backTestData = getPercentages(backTestData, money)
    
    backTestData = buyAndSell(backTestData, money, weight)

    return backTestData
    
def generateWeightedSignals(funcData):
    """
    Generate weighted trading signals by combining all individual indicator signals.
    
    Args:
        funcData (pandas.DataFrame): DataFrame containing individual signal columns (SigMA, SigRSI, etc.)
    
    Returns:
        None: Modifies funcData in place by adding 'Signals' column
    """
    signals = {}
    weightedSignals = []
    tempData = funcData
    
    # Find all signal columns (columns that start with 'Sig')
    signal_columns = [col for col in tempData.columns if col.startswith('Sig')]
    
    if not signal_columns:
        print("Warning: No signal columns found. Creating default signals of 0.")
        tempData['Signals'] = [0] * len(tempData)
        return
    
    # Extract signal data from each column
    for header in signal_columns:
        signals[header] = tempData[header].values.tolist()
    
    # Combine all signals by summing them
    for header in signals.keys():
        if len(weightedSignals) < 1:
            weightedSignals = signals[header].copy()  # Use copy to avoid reference issues
        else:
            for i in range(len(signals[header])):
                weightedSignals[i] = weightedSignals[i] + signals[header][i]
    
    print(f"Generated weighted signals from {len(signal_columns)} indicators: {signal_columns}")
    print(f"Signal range: {min(weightedSignals):.2f} to {max(weightedSignals):.2f}\n")
    
    tempData['Signals'] = weightedSignals

def generateSignals(funcData, method, parameters):
    
    '''
    A method generates signals based on indicators. These signals imply buy sell or hold.
    '''
    
    useMethod = method(funcData, parameters)
    
    generateWeightedSignals(useMethod)
    
    return useMethod 

def predictions(funcData, method, money, tolerance, parameters):
    print("Simulating financial data...")
    print("")
    
    forecastData = funcData.copy()
    
    forecastedPrices = Forecast.forecastPricesHighLowVolume(forecastData, 200, 12)
    
    print("Simulation saved. \n")
    
    print(f"Testing {method} on simulation... \n")
    
    forecastedPricesWithSignals = generateSignals(forecastedPrices, method, parameters)

    simulation = backTest(forecastedPricesWithSignals, money, tolerance)

    print("Simulation tested.")
    
    return simulation

def setParameters():
    return 20, 40

def outputToFiles(funcData, predict, symbol, end_date):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(current_dir, f"Analysis_of_{symbol}")
    os.makedirs(output_dir, exist_ok=True)

    predictionsOutputFile = os.path.join(output_dir, f"trade_predictions_{symbol}_{end_date}.txt")
    dataOutputFile = os.path.join(output_dir, f"data_used_{symbol}_{end_date}.txt")

    with open(predictionsOutputFile, "w") as f:
        f.write(predict.to_string())

    with open(dataOutputFile, "w") as f:
        f.write(funcData.to_string())
        
    return predictionsOutputFile, dataOutputFile

def getDates(start, end):
    
    '''
    0 returns today, a number gives the number of years ago to start the data from
    '''
    
    if end == 0:
        end_date = datetime.today().strftime('%Y-%m-%d')
    else:
        print("Additional time commands soon.")
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    start_date = (datetime.today() - relativedelta(years=start)).strftime('%Y-%m-%d')

    return start_date, end_date

def getPercentages(funcData, money):
    """
    Calculate percentage returns and cumulative performance metrics.
    
    Args:
        funcData (pandas.DataFrame): Stock data with 'Close' column
        money (float): Initial investment amount
    
    Returns:
        pandas.DataFrame: Data with added percentage return columns
    """
    percentData = funcData.copy()
    
    # Get close prices as a pandas Series
    close_prices = percentData['Close']
    
    # Calculate daily percentage returns (price[t] / price[t-1])
    # This gives us the daily return factor (1.05 = 5% gain, 0.95 = 5% loss)
    daily_returns = close_prices.pct_change() + 1  # pct_change() gives (p[t]-p[t-1])/p[t-1], so add 1
    daily_returns.iloc[0] = 1  # Set first day return to 1 (no change from non-existent previous day)
    
    percentData['%Diff'] = daily_returns
    
    # Calculate cumulative returns (compound growth)
    cumulative_returns = daily_returns.cumprod()
    percentData['%TotDiff'] = cumulative_returns
    
    # Calculate portfolio value over time (buy and hold strategy)
    portfolio_values = money * cumulative_returns
    percentData['Portfolio'] = portfolio_values
    
    print(f"Performance calculation complete:")
    print(f"  Initial investment: ${money:,.2f}")
    print(f"  Final portfolio value: ${portfolio_values.iloc[-1]:,.2f}")
    print(f"  Total return: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
    
    return percentData

def getTolerance():
    '''
    Tolerance levels:
    
        1: Super conservative - very hard to buy and sell
        2: Conservative       - hard to buy and sell
        3: Balanced           - no changes in weighting towards buying or selling
        4: Risky              - open to accept bigger losses for bigger gains
        5: Very risky         - trying to gamble on making lots of money
        
    '''
    tolerance = 3

    return tolerance

def main():
    tolerance = getTolerance()
    method = method1 # Method by which we develop signals
    parameters = {'shortWindow': 10, 'longWindow': 20, 'RSIWindow': 14, 'bollingerBands' : 20, 'ATR' : 14, 'stochasticOscillator' : 14}
    symbol = getSymbol()
    
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
    print("Generating signals... \n")

    testData = generateSignals(data, method, parameters)
    
    money = 1000
    
    historicalTotalReturns = getPercentages(data, money)
    print(f"Back testing data using {method}... \n")
    strategyTotalReturns = backTest(testData, money, tolerance)
    predict = predictions(data, method, money, tolerance, parameters)

    outputToFiles(data, predict, symbol, end_date)    
    
    print("")
    print("")
    while True:
        choice0 = input("See Historical and back test data (y/n)? \n")
        if choice0 == "y":
            print("Historical Returns")        
            print(historicalTotalReturns, "\n \n")
            print("Strategic Returns")
            print(strategyTotalReturns, "\n \n")
            break
        elif choice0 == "n":
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            
    print(f"Predictions saved to trade_predictions_{symbol}_{end_date}.")
    print(f"Analysis used saved to analysis_used_{symbol}_{end_date}.")
    print(f"Data used saved to data_used_{symbol}_{end_date}. \n \n")
    print(f"Historically, returns from {start_date} to {end_date} is {historicalTotalReturns['%TotDiff'].iloc[-1]}")
    print(f"Using the strategy, returns from {start_date} to {end_date} is {strategyTotalReturns['%TotDiff'].iloc[-1]} \n \n")
    
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
