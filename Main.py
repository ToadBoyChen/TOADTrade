from datetime import datetime
import yfinance as yf
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
    data = yf.download(symbol, start=start_date, end=end_date)
    isEmpty(data)
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
    '''
    Backtest a simple trading strategy with buying and selling signals.
    
    - Signals >= weight: Buy
    - Signals <= -weight: Sell
    - Otherwise: Hold
    '''
    workingData = funcData
    strategyPortfolio = [0]
    signals = workingData['Signals'].values.tolist()
    stockGainLoss = workingData['%Diff'].values.tolist()

    buy_signals = sum(1 for signal in signals if signal >= weight)

    purchasingPower = money / buy_signals if buy_signals > 0 else 0
    sellingPower = purchasingPower
    totalSpent = [0]

    for i in range(1, len(signals)):
        if signals[i] >= weight:
            totalSpent.append(totalSpent[-1] + purchasingPower)
            strategyPortfolio.append(strategyPortfolio[i - 1] + purchasingPower * (stockGainLoss[i]))
        elif signals[i] <= -weight:
            if strategyPortfolio[i - 1] - sellingPower >= 0:
                totalSpent.append(totalSpent[-1])
                strategyPortfolio.append(strategyPortfolio[i - 1] - (sellingPower * (stockGainLoss[i])))
            else:
                totalSpent.append(totalSpent[-1])
                strategyPortfolio.append(strategyPortfolio[i - 1])
        else:
            totalSpent.append(totalSpent[-1])
            strategyPortfolio.append(strategyPortfolio[i - 1])

    workingData['TSpent'] = totalSpent
    workingData['AlgoPort'] = strategyPortfolio

    print(f"Total spent on buying: {totalSpent[-1]}, Total value of portfolio after trading: {strategyPortfolio[-1]} \n")

    return workingData


    
def backTest(funcData, money, tolerance):
    backTestData = funcData.copy()

    weight = tolerance/3
    
    backTestData = getPercentages(backTestData, money)
    
    backTestData = buyAndSell(backTestData, money, weight)

    return backTestData
    
def generateWeightedSignals(funcData):
    signals = {}
    
    weightedSignals = []
    
    tempData = funcData
    
    for header in tempData.columns.get_level_values(0):
        if "Sig" in header:
            signals[header] = tempData[header].values.tolist()  

    for header in signals.keys():
        if len(weightedSignals) < 1:
            weightedSignals = signals[header]
        else:
            for i in range(0, len(signals[header])):
                weightedSignals[i] = weightedSignals[i] + signals[header][i]
          
    print("Generating weighted signals... \n")  
    # print(weightedSignals, "\n")
    
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
    
    forecastedPrices = Forecast.forecastPrices(forecastData, 200)
    
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
    percentData = funcData.copy()
    try:
        close = [x[0] for x in percentData['Close'].values.tolist()]
    except:
        close = percentData['Close']
    
    percentReturns = []
    
    try:
        for i in range(1, len(close)):
            percentReturns.append(close.iloc[i]/close.iloc[i-1])
    except:
         for i in range(1, len(close)):
            percentReturns.append(close[i]/close[i-1])       
        
    percentReturns.insert(0, 1)    
    percentData['%Diff'] = percentReturns
    
    cumulativePercentageReturns = [1]
    for i in range(1, len(percentReturns)):
        cumulativePercentageReturns.append(cumulativePercentageReturns[-1] * percentReturns[i])
        
    percentData['%TotDiff'] = cumulativePercentageReturns
    
    portfolioValue = [money]
    
    for i in range(1, len(percentReturns)):
        portfolioValue.append(portfolioValue[-1] * percentReturns[i])
        
    percentData['Portfolio'] = portfolioValue

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
    
    print("")
    
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
    
    print("")
    
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
