import sys
import os
import re
import pytest
import pandas as pd # For checking DataFrame results if Main.main is adapted

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Main # Main.py

# Default parameters
default_params = {
    "shortWindow": 10,
    "longWindow": 20,
    "RSIWindow": 14,
    "bollingerBands": 20,
    "ATR": 14,
    "stochasticOscillator": 14
}

def profitability_ok(perf_factor, min_return=0.0, max_return=5.0):
    """
    Tighten profitability range for stricter testing.
    - min_return is minimum acceptable factor (e.g., 0.0 for no loss)
    - max_return is maximum acceptable factor (e.g., 5 for 400% gain)
    Returns True if in range, False otherwise.
    """
    return min_return <= perf_factor <= max_return

# Fixture to run the pipeline with default params once and reuse the result
@pytest.fixture(scope="module") # module scope: runs once per test module
def default_pipeline_run(monkeypatch_module, capsys_module):
    pass

def run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1", expect_success=True):
    inputs = iter([symbol, years])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    result_metric = Main.main(params, test_mode=True)
    output = capsys.readouterr().out
    if expect_success:
        assert "Trading Strategy Results:" in output, f"Expected 'Trading Strategy Results:' in output. Got: {output}"
        assert "Total return:" in output, f"Expected 'Total return:' in output. Got: {output}"
        match = re.search(r"Total return:\s*([-+]?[0-9]*\.?[0-9]+)%", output)
        assert match, f"Could not parse total return from output. Got: {output}"
        assert isinstance(result_metric, float), f"Expected float return from Main.main, got {result_metric}"
        return result_metric
    else:
        return output

def test_pipeline_runs_default_params(monkeypatch, capsys):
    perf_factor = run_pipeline_with_params(monkeypatch, capsys, default_params)
    assert isinstance(perf_factor, float)
    assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

@pytest.mark.parametrize("symbol, years, expected_min_days", [
    ("AAPL", "1", 240),  # Reduce expected days to account for holidays and weekends
])
def test_pipeline_fetches_sufficient_data(monkeypatch, capsys, symbol, years, expected_min_days):
    inputs = iter([symbol, years])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    result_metric = Main.main(default_params, test_mode=True)
    output = capsys.readouterr().out
    
    # Check that we got a valid result
    assert isinstance(result_metric, float)
    
    # Check for the data fetching message
    match_days = re.search(r"Successfully fetched (\d+) days of data", output)
    assert match_days, f"Could not find number of days fetched in output: {output}"
    days_fetched = int(match_days.group(1))
    assert days_fetched >= expected_min_days, f"Expected at least {expected_min_days} days for {symbol} over {years} years, got {days_fetched}."
    
    # Check profitability
    assert profitability_ok(result_metric), f"Profitability out of bounds: got {result_metric}"

@pytest.mark.parametrize("param_key, alt_value", [
    ("shortWindow", 5),
    ("RSIWindow", 7),
])
def test_parameter_variations(monkeypatch, capsys, param_key, alt_value):
    params_alt = default_params.copy()
    params_alt[param_key] = alt_value
    perf_factor = run_pipeline_with_params(monkeypatch, capsys, params_alt, symbol="AAPL", years="1")
    assert isinstance(perf_factor, float), f"Pipeline with {param_key}={alt_value} did not return a float."
    assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

@pytest.mark.parametrize("indicator_function, params", [
    (Main.Indicators.computeMovingAverages, {"shortWindow": 10, "longWindow": 20}),
    (Main.Indicators.computeRSI, {"window": 14}),
])
def test_indicator_signal_quality(indicator_function, params):
    data = pd.DataFrame({
        "Close": [100, 102, 101, 105, 107, 110],
        "High": [101, 103, 102, 106, 108, 111],
        "Low": [99, 100, 100, 104, 106, 109],
        "Volume": [1000, 1200, 1100, 1300, 1400, 1500],
    })
    processed_data = indicator_function(data, **params)
    signal_column = [col for col in processed_data.columns if col.startswith("Sig")][0]
    signals = processed_data[signal_column]
    assert signals.min() >= -1, f"Signal values below expected range: {signals.min()}"
    assert signals.max() <= 1, f"Signal values above expected range: {signals.max()}"

@pytest.mark.parametrize("params", [
    {"shortWindow": 5, "longWindow": 10, "RSIWindow": 7, "bollingerBands": 10, "ATR": 7, "stochasticOscillator": 7},
])
def test_pipeline_with_custom_parameters(monkeypatch, capsys, params):
    perf_factor = run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1")
    assert isinstance(perf_factor, float), "Expected numeric return value."
    assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"
