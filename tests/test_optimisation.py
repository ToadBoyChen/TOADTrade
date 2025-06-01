import sys
import os
import re
import pytest

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Main

# Default parameters
default_params = {
    "shortWindow": 10,
    "longWindow": 20,
    "RSIWindow": 14,
    "bollingerBands": 20,
    "ATR": 14,
    "stochasticOscillator": 14
}

def run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1"):
    """Helper function to run pipeline with params."""
    # Enough inputs for both rounds of input prompts in Main.main
    # The final "1" is so the exit menu picks "Exit" and doesn't loop forever
    inputs = iter([symbol, years, "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    Main.main(params, test_mode=True)  # pass params and test_mode so it doesn't re-prompt at the end

    output = capsys.readouterr().out
    assert "Total return:" in output, "Expected 'Total return:' in output."

    match = re.search(r"Total return:\s*([-+]?[0-9]*\.?[0-9]+)%", output)
    assert match, "Could not parse total return from output."
    return float(match.group(1))

def test_pipeline_runs_default_params(monkeypatch, capsys):
    """Test pipeline runs with default params."""
    perf = run_pipeline_with_params(monkeypatch, capsys, default_params)
    assert isinstance(perf, float)

@pytest.mark.parametrize("param, alt_value", [
    ("shortWindow", 5),
    ("RSIWindow", 7),
    ("bollingerBands", 10),
    ("ATR", 7),
    ("stochasticOscillator", 7)
])
def test_higher_sensitivity_params_yield_higher_returns(monkeypatch, capsys, param, alt_value):
    """Test parameter sensitivity impact."""
    params_alt = default_params.copy()
    params_alt[param] = alt_value

    perf_default = run_pipeline_with_params(monkeypatch, capsys, default_params)
    perf_alt = run_pipeline_with_params(monkeypatch, capsys, params_alt)

    print(f"{param}: Default={perf_default:.2f}%, Alt={perf_alt:.2f}%")
    # This asserts the "sensitive" parameter didn't degrade performance
    assert perf_alt >= perf_default, f"{param}={alt_value} underperformed"

def test_pipeline_handles_invalid_input(monkeypatch, capsys):
    """Test error handling."""
    # Enough "1"s to get through all prompts
    inputs = iter(["invalid_symbol", "1", "1"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    # Call with params and test_mode just to be safe (even though params don't matter for invalid symbol)
    Main.main(default_params, test_mode=True)
    output = capsys.readouterr().out
    assert "Error" in output or "Invalid" in output