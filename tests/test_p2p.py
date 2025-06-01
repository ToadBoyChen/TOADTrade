import sys
import os
import pytest
import Main

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
default_params = {
    "shortWindow": 10,
    "longWindow": 20,
    "RSIWindow": 14,
    "bollingerBands": 20,
    "ATR": 14,
    "stochasticOscillator": 14
}

def test_signals_to_backtest_consistency(monkeypatch):
    """
    P2P: Ensure that signals generated are valid input for backTest,
    and that the final AlgoPort value matches the expected portfolio calculation.
    """
    # Simulate user input for symbol and years
    inputs = iter(["AAPL", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    # Fetch data and generate signals
    symbol = "AAPL"
    years = 1
    start_date, end_date = Main.getDates(years, 0)
    data = Main.getData(symbol, start_date, end_date)
    signals = Main.generateSignals(data, Main.method1, default_params)
    # Run backtest
    backtest_result = Main.backTest(signals, 1000, 3)
    # P2P check: AlgoPort should be close to Portfolio at the end (if logic is consistent)
    assert abs(backtest_result['AlgoPort'].iloc[-1] - backtest_result['Portfolio'].iloc[-1]) < 1e-2

def test_strategy_vs_buy_and_hold(monkeypatch):
    """
    P2P: Ensure that the strategy backtest and buy-and-hold produce different results,
    and both are positive (profitable) for a reasonable market period.
    """
    # Simulate user input for symbol and years
    inputs = iter(["AAPL", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    symbol = "AAPL"
    years = 1
    start_date, end_date = Main.getDates(years, 0)
    data = Main.getData(symbol, start_date, end_date)
    buy_and_hold = Main.getPercentages(data, 1000)
    signals = Main.generateSignals(data, Main.method1, default_params)
    backtest_result = Main.backTest(signals, 1000, 3)
    # P2P check: The two approaches should not be identical
    assert not buy_and_hold['Portfolio'].equals(backtest_result['AlgoPort'])
    # Both should be profitable (final value > initial)
    assert buy_and_hold['Portfolio'].iloc[-1] > 1000
    assert backtest_result['AlgoPort'].iloc[-1] > 1000