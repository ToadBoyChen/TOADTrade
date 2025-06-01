from datetime import datetime
import yfinance as yf
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
import Indicators
import Forecast
import json
import os

def save_optimized_parameters(symbol, params):
    """Save optimized parameters for a stock to a JSON file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(current_dir, "optimized_params")
    os.makedirs(params_dir, exist_ok=True)
    
    params_file = os.path.join(params_dir, f"{symbol}_params.json")
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Optimized parameters saved to {params_file}")

def load_optimized_parameters(symbol):
    """Load optimized parameters for a stock from a JSON file if available"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(current_dir, "optimized_params", f"{symbol}_params.json")
    
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded optimized parameters for {symbol}")
        return params
    
    return None

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

def get_analysis_period():
    """Get analysis period with validation"""
    while True:
        try:
            start = int(input("When would you like to test from (1 - 1 year ago, 2 - 2 years ago, ...)? "))
            if start >= 1:
                return start
            print("Cannot accept negatives")
        except ValueError:
            print("Input not an integer, choose again")

def handle_user_choice(symbol):
    """Handle menu choices without recursion"""
    while True:
        choice = input("Exit (1), work with data (2) or try again (3): ").strip()
        if choice == "1":
            return None
        elif choice == "2":
            workWithData(symbol)
            continue
        elif choice == "3":
            return "restart"
        print("Invalid input. Please enter 1, 2, or 3.")
        
import itertools
import concurrent.futures

def optimize_parameters(symbol, start_date, end_date, money=1000, test_mode=False):
    """
    Optimize trading strategy parameters for a specific stock.
    
    Args:
        symbol (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        money (float): Initial capital for backtesting
        test_mode (bool): Whether to run in test mode
        
    Returns:
        dict: Optimized parameters
        float: Best performance metric
    """
    print(f"\nOptimizing parameters for {symbol}...")
    
    # Get historical data
    data = getData(symbol, start_date, end_date)
    
    # Define parameter ranges to test
    param_grid = {
        'shortWindow': [5, 10, 15, 20],
        'longWindow': [20, 30, 40, 50],
        'RSIWindow': [7, 14, 21],
        'bollingerBands': [15, 20, 25],
        'ATR': [7, 14, 21],
        'stochasticOscillator': [7, 14, 21]
    }
    
    # Ensure longWindow is always greater than shortWindow
    param_combinations = []
    for short in param_grid['shortWindow']:
        for long in param_grid['longWindow']:
            if long > short:
                for rsi in param_grid['RSIWindow']:
                    for bb in param_grid['bollingerBands']:
                        for atr in param_grid['ATR']:
                            for stoch in param_grid['stochasticOscillator']:
                                param_combinations.append({
                                    'shortWindow': short,
                                    'longWindow': long,
                                    'RSIWindow': rsi,
                                    'bollingerBands': bb,
                                    'ATR': atr,
                                    'stochasticOscillator': stoch
                                })
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Function to test a single parameter combination
    def test_params(params):
        try:
            # Generate signals with the current parameters
            test_data = generateSignals(data.copy(), method1, params)
            
            # Run backtest with a fixed tolerance
            tolerance = 3  # Using balanced tolerance
            backtest_results = backTest(test_data, money, tolerance)
            
            # Get final portfolio value as performance metric
            final_value = backtest_results['AlgoPort'].iloc[-1]
            return params, final_value
        except Exception as e:
            print(f"Error testing parameters: {e}")
            return params, 0
    
    # Use parallel processing to speed up optimization
    best_params = None
    best_performance = 0
    
    # Progress tracking
    total_combinations = len(param_combinations)
    completed = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_params = {executor.submit(test_params, params): params for params in param_combinations}
        
        for future in concurrent.futures.as_completed(future_to_params):
            params, performance = future.result()
            completed += 1
            
            # Print progress every 10%
            if completed % max(1, total_combinations // 10) == 0:
                print(f"Progress: {completed}/{total_combinations} combinations tested ({completed/total_combinations*100:.1f}%)")
            
            if performance > best_performance:
                best_performance = performance
                best_params = params
    
    initial_performance = money
    percent_improvement = ((best_performance - initial_performance) / initial_performance) * 100
    
    print(f"\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best performance: ${best_performance:.2f} (ROI: {percent_improvement:.2f}%)")
    
    return best_params, best_performance

def display_results(historicalTotalReturns, strategyTotalReturns, symbol, start_date, end_date):
    """Display comparison of buy-and-hold vs. strategy performance"""
    buy_hold_final = historicalTotalReturns['Portfolio'].iloc[-1]
    strategy_final = strategyTotalReturns['AlgoPort'].iloc[-1]
    initial_investment = historicalTotalReturns['Portfolio'].iloc[0]
    
    buy_hold_return = ((buy_hold_final / initial_investment) - 1) * 100
    strategy_return = ((strategy_final / initial_investment) - 1) * 100
    
    print("\n" + "="*50)
    print(f"PERFORMANCE SUMMARY FOR {symbol}: {start_date} to {end_date}")
    print("="*50)
    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"Buy & Hold strategy: ${buy_hold_final:.2f} (Return: {buy_hold_return:.2f}%)")
    print(f"Trading strategy: ${strategy_final:.2f} (Return: {strategy_return:.2f}%)")
    
    outperformance = strategy_return - buy_hold_return
    if outperformance > 0:
        print(f"Strategy OUTPERFORMED Buy & Hold by {outperformance:.2f}%")
    else:
        print(f"Strategy UNDERPERFORMED Buy & Hold by {abs(outperformance):.2f}%")
    print("="*50)

def main(params=None, test_mode=False):
    """Main program loop with parameter optimization and caching"""
    tolerance = getTolerance()
    method = method1
    
    symbol = getSymbol()
    start = get_analysis_period()
    
    start_date, end_date = getDates(start, 0)
    
    # If parameters aren't provided, try to load saved ones or optimize
    if params is None:
        params = load_optimized_parameters(symbol)
        
        if params is None:
            print(f"No saved parameters found for {symbol}. Optimizing...")
            params, _ = optimize_parameters(symbol, start_date, end_date, money=1000, test_mode=test_mode)
            save_optimized_parameters(symbol, params)
        else:
            print(f"Using saved parameters for {symbol}: {params}")
            
            # Ask if user wants to re-optimize
            reoptimize = input("Would you like to re-optimize parameters? (y/n): ").strip().lower()
            if reoptimize == 'y':
                params, _ = optimize_parameters(symbol, start_date, end_date, money=1000, test_mode=test_mode)
                save_optimized_parameters(symbol, params)
    else:
        print("\nUsing provided parameters:", params)
    
    # Get data and run strategy with optimized parameters
    data = getData(symbol, start_date, end_date)
    testData = generateSignals(data, method, params)
    money = 1000
    
    historicalTotalReturns = getPercentages(data, money)
    strategyTotalReturns = backTest(testData, money, tolerance)
    
    # For test mode, return early with just the returns
    if test_mode:
        return float(strategyTotalReturns['%TotDiff'].iloc[-1])
        
    predict = predictions(data, method, money, tolerance, params)
    outputToFiles(data, predict, symbol, end_date)
    
    display_results(historicalTotalReturns, strategyTotalReturns, symbol, start_date, end_date)
    
    return handle_user_choice(symbol)

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
