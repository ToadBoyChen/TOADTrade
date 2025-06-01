import sys, os, re
import pytest

# Add repository root to path (assumes tests/ is under repo root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Main  # Core trading pipeline
import Indicators  # Technical indicators module
import Forecast    # Forecasting module

# Default parameters from TOADTrade README (shortWindow=10, longWindow=20, RSI=14, BB=20, ATR=14, Stochastic=14):contentReference[oaicite:1]{index=1}.
default_params = {
    "shortWindow": 10,
    "longWindow": 20,
    "RSIWindow": 14,
    "bollingerBands": 20,
    "ATR": 14,
    "stochasticOscillator": 14
}

def run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1"):
    """
    Helper to run the Main pipeline with given parameters and return the total return percentage.
    Mocks user input for symbol and period, sets the strategy parameters, and captures output.
    """
    # Set parameters in Main (assumes Main reads Main.parameters in main())
    Main.parameters = params

    # Mock inputs: stock symbol and analysis period (years)
    inputs = iter([symbol, years])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    # Run the main pipeline (should include data fetch, indicators, backtest, forecast)
    Main.main()

    # Capture printed output and extract the total return (%):contentReference[oaicite:2]{index=2}
    output = capsys.readouterr().out
    assert "Total return:" in output, "Expected 'Total return' in output."

    # Parse the percentage value after 'Total return:'
    match = re.search(r"Total return:\s*([-+]?[0-9]*\.?[0-9]+)%", output)
    assert match, "Could not parse total return from output."
    return float(match.group(1))

def test_pipeline_runs_default_params(monkeypatch, capsys):
    """
    Confirm that the trading pipeline runs without error using default parameters.
    """
    perf = run_pipeline_with_params(monkeypatch, capsys, default_params)
    # Output should indicate some return (could be positive or negative)
    assert isinstance(perf, float)

import pytest
from unittest.mock import MagicMock

# Mock the Main module to isolate the pipeline's behavior
Main = MagicMock()
Main.main = MagicMock(return_value=None)

# Define a test fixture to set up the pipeline's parameters
@pytest.fixture
def pipeline_params():
    return {
        "shortWindow": 10,
        "longWindow": 20,
        "RSIWindow": 14,
        "bollingerBands": 20,
        "ATR": 14,
        "stochasticOscillator": 14
    }

# Test the pipeline's behavior with default parameters
def test_pipeline_runs_default_params(monkeypatch, capsys, pipeline_params):
    # Set up the pipeline's parameters
    Main.parameters = pipeline_params

    # Mock the user input
    inputs = iter(["AAPL", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    # Run the pipeline
    Main.main()

    # Capture the output
    output = capsys.readouterr().out

    # Assert that the pipeline ran without errors
    assert "Total return:" in output

# Test the pipeline's behavior with varying parameters
@pytest.mark.parametrize("param, alt_value", [
    ("shortWindow", 5),
    ("RSIWindow", 7),
    ("bollingerBands", 10),
    ("ATR", 7),
    ("stochasticOscillator", 7)
])
def test_pipeline_runs_with_varying_params(monkeypatch, capsys, pipeline_params, param, alt_value):
    # Set up the pipeline's parameters
    params = pipeline_params.copy()
    params[param] = alt_value

    # Mock the user input
    inputs = iter(["AAPL", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    # Run the pipeline
    Main.main()

    # Capture the output
    output = capsys.readouterr().out

    # Assert that the pipeline ran without errors
    assert "Total return:" in output

# Test the pipeline's error handling
def test_pipeline_handles_invalid_input(monkeypatch, capsys):
    # Mock the user input with invalid data
    inputs = iter([" invalid", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    # Run the pipeline
    Main.main()

    # Capture the output
    output = capsys.readouterr().out

    # Assert that the pipeline handled the error correctly
    assert "Error:" in output@pytest.mark.parametrize("param, alt_value", [
    ("shortWindow", 5),          # shorter MA window (more sensitive):contentReference[oaicite:3]{index=3}
    ("RSIWindow", 7),            # shorter RSI window:contentReference[oaicite:4]{index=4}
    ("bollingerBands", 10),      # shorter Bollinger window:contentReference[oaicite:5]{index=5}
    ("ATR", 7),                  # shorter ATR window:contentReference[oaicite:6]{index=6}
    ("stochasticOscillator", 7)  # shorter Stochastic %K window:contentReference[oaicite:7]{index=7}
])
def test_higher_sensitivity_params_yield_higher_returns(monkeypatch, capsys, param, alt_value):
    """
    For each listed parameter, compare default vs a more sensitive setting (smaller window).
    We expect the more sensitive (smaller window) setting to yield >= returns than default.
    """
    # Prepare parameter sets
    params_default = default_params.copy()
    params_alt = default_params.copy()
    params_alt[param] = alt_value

    # Run pipeline with default parameters
    perf_default = run_pipeline_with_params(monkeypatch, capsys, params_default)
    # Run pipeline with altered (more sensitive) parameter
    perf_alt = run_pipeline_with_params(monkeypatch, capsys, params_alt)

    # Log both performances
    print(f"Param {param}: default return = {perf_default:.2f}%, alt return = {perf_alt:.2f}%")

    # Assert that the more sensitive setting does not underperform default:contentReference[oaicite:8]{index=8}.
    # (Assumes that a more reactive parameter yields at least comparable performance.)
    assert perf_alt >= perf_default, (
        f"Expected higher or equal return with {param}={alt_value}, "
        f"got {perf_alt:.2f}% vs {perf_default:.2f}%"
    )
