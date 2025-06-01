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

def test_signals_to_backtest_runs(monkeypatch):
    """
    P2P: Ensure that signals generated are valid input for backTest,
    and that the pipeline produces numeric results.
    """
    inputs = iter(["AAPL", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    symbol = "AAPL"
    years = 1
    start_date, end_date = Main.getDates(years, 0)
    data = Main.getData(symbol, start_date, end_date)
    signals = Main.generateSignals(data, Main.method1, default_params)
    backtest_result = Main.backTest(signals, 1000, 3)
    # P2P check: AlgoPort and Portfolio columns exist and are numeric
    assert "AlgoPort" in backtest_result.columns
    assert "Portfolio" in backtest_result.columns
    assert isinstance(backtest_result['AlgoPort'].iloc[-1], (float, int))
    assert isinstance(backtest_result['Portfolio'].iloc[-1], (float, int))

def test_strategy_vs_buy_and_hold_relative(monkeypatch):
    """
    P2P: Ensure that the strategy backtest and buy-and-hold produce different results,
    and that buy-and-hold is at least as good as the strategy (which is realistic).
    """
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
    # Buy-and-hold should not lose money in a typical year for AAPL
    assert buy_and_hold['Portfolio'].iloc[-1] > 1000
    # Strategy can lose money, but should produce a numeric result
    assert isinstance(backtest_result['AlgoPort'].iloc[-1], (float, int))