# import sys
# import os
# import re
# import pytest
# import pandas as pd # For checking DataFrame results if Main.main is adapted

# # Add repository root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import Main # Main.py

# # Default parameters
# default_params = {
#     "shortWindow": 10,
#     "longWindow": 20,
#     "RSIWindow": 14,
#     "bollingerBands": 20,
#     "ATR": 14,
#     "stochasticOscillator": 14
# }

# def profitability_ok(perf_factor, min_return=0.0, max_return=5.0):
#     """
#     Tighten profitability range for stricter testing.
#     - min_return is minimum acceptable factor (e.g., 0.0 for no loss)
#     - max_return is maximum acceptable factor (e.g., 5 for 400% gain)
#     Returns True if in range, False otherwise.
#     """
#     return min_return <= perf_factor <= max_return

# # Fixture to run the pipeline with default params once and reuse the result
# @pytest.fixture(scope="module") # module scope: runs once per test module
# def default_pipeline_run(monkeypatch_module, capsys_module):
#     pass

# def run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1", expect_success=True):
#     inputs = iter([symbol, years])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     result_metric = Main.main(params, test_mode=True)
#     output = capsys.readouterr().out
#     if expect_success:
#         assert "Trading Strategy Results:" in output, f"Expected 'Trading Strategy Results:' in output. Got: {output}"
#         assert "Total return:" in output, f"Expected 'Total return:' in output. Got: {output}"
#         match = re.search(r"Total return:\s*([-+]?[0-9]*\.?[0-9]+)%", output)
#         assert match, f"Could not parse total return from output. Got: {output}"
#         assert isinstance(result_metric, float), f"Expected float return from Main.main, got {result_metric}"
#         return result_metric
#     else:
#         return output

# def test_pipeline_runs_default_params(monkeypatch, capsys):
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, default_params)
#     assert isinstance(perf_factor, float)
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("param_key, alt_value, expected_behavior_description", [
#     ("shortWindow", 5, "Shorter window, potentially more trades"),
#     ("RSIWindow", 7, "Shorter RSI, more sensitive"),
#     ("bollingerBands", 10, "Shorter Bollinger, bands closer to price"),
#     ("ATR", 7, "Shorter ATR, more reactive volatility"),
#     ("stochasticOscillator", 7, "Shorter Stochastic, faster oscillator"),
#     ("shortWindow", 50, "Short window longer than default long window"),
# ])
# def test_parameter_variations(monkeypatch, capsys, param_key, alt_value, expected_behavior_description):
#     print(f"Testing: {param_key}={alt_value} ({expected_behavior_description})")
#     perf_default_factor = run_pipeline_with_params(monkeypatch, capsys, default_params, symbol="AAPL", years="1")
#     params_alt = default_params.copy()
#     params_alt[param_key] = alt_value
#     if param_key == "shortWindow" and alt_value > params_alt["longWindow"]:
#         print(f"Note: Testing with shortWindow ({alt_value}) > longWindow ({params_alt['longWindow']})")
#     perf_alt_factor = run_pipeline_with_params(monkeypatch, capsys, params_alt, symbol="AAPL", years="1")
#     assert isinstance(perf_alt_factor, float), f"Pipeline with {param_key}={alt_value} did not return a float."
#     assert profitability_ok(perf_alt_factor), f"Profitability out of bounds: got {perf_alt_factor}"
#     print(f"Performance with {param_key}={alt_value}: {perf_alt_factor-1:.2%}. Default performance: {perf_default_factor-1:.2%}")

# def test_pipeline_handles_invalid_symbol_then_valid(monkeypatch, capsys):
#     inputs = iter(["INVALID_TICKER", "AAPL", "1"])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     final_perf_factor = Main.main(default_params, test_mode=True)
#     output = capsys.readouterr().out
#     assert "Invalid symbol: INVALID_TICKER. Please try again." in output, \
#         f"Expected specific error for INVALID_TICKER. Output: {output}"
#     assert "Successfully fetched data for AAPL." in output, \
#         f"Expected successful fetch for AAPL after invalid. Output: {output}"
#     assert "Trading Strategy Results:" in output, \
#         f"Expected trading results for AAPL. Output: {output}"
#     assert isinstance(final_perf_factor, float), "Expected a float performance for the final valid run."
#     assert profitability_ok(final_perf_factor), f"Profitability out of bounds: got {final_perf_factor}"

# @pytest.mark.parametrize("symbol, years, expected_min_days", [
#     ("AAPL", "1", 252),  # Increase minimum days for stricter validation
#     ("TSLA", "2", 504),
#     ("MSFT", "5", 1260),
# ])
# def test_pipeline_fetches_sufficient_data(monkeypatch, capsys, symbol, years, expected_min_days):
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, default_params, symbol=symbol, years=years)
#     assert isinstance(perf_factor, float)
#     output = capsys.readouterr().out
#     match_days = re.search(r"Successfully fetched (\d+) days of data", output)
#     assert match_days, "Could not find number of days fetched."
#     days_fetched = int(match_days.group(1))
#     assert days_fetched >= expected_min_days, f"Expected at least {expected_min_days} days for {symbol} over {years} years, got {days_fetched}."
#     assert perf_factor > 1.0, f"Expected performance factor > 1.0 for {symbol}, got {perf_factor}."
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("symbol, years, expected_max_days", [
#     ("AAPL", "1", 260),  # Keep maximum days strict
#     ("TSLA", "2", 520),
#     ("MSFT", "5", 1300),
# ])
# def test_pipeline_fetches_no_excessive_data(monkeypatch, capsys, symbol, years, expected_max_days):
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, default_params, symbol=symbol, years=years)
#     assert isinstance(perf_factor, float)
#     output = capsys.readouterr().out
#     match_days = re.search(r"Successfully fetched (\d+) days of data", output)
#     assert match_days, "Could not find number of days fetched."
#     days_fetched = int(match_days.group(1))
#     assert days_fetched <= expected_max_days, f"Expected at most {expected_max_days} days for {symbol} over {years} years, got {days_fetched}."
#     assert perf_factor > 1.0, f"Expected performance factor > 1.0 for {symbol}, got {perf_factor}."
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("param_key, extreme_value, expected_min_return", [
#     ("shortWindow", 1, 0.0),  # Tighten expected minimum return
#     ("longWindow", 1000, 0.0),
#     ("RSIWindow", 1, 0.0),
#     ("bollingerBands", 1, 0.0),
# ])
# def test_pipeline_extreme_parameter_values(monkeypatch, capsys, param_key, extreme_value, expected_min_return):
#     params_extreme = default_params.copy()
#     params_extreme[param_key] = extreme_value
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, params_extreme, symbol="AAPL", years="1")
#     assert isinstance(perf_factor, float), f"Pipeline with {param_key}={extreme_value} did not return a float."
#     assert perf_factor >= expected_min_return, f"Expected performance factor >= {expected_min_return} for {param_key}={extreme_value}, got {perf_factor}."
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("param_key, extreme_value, expected_max_return", [
#     ("shortWindow", 1, 3.0),  # Tighten expected maximum return
#     ("longWindow", 1000, 3.0),
#     ("RSIWindow", 1, 3.0),
#     ("bollingerBands", 1, 3.0),
# ])
# def test_pipeline_extreme_parameter_values_high_volatility(monkeypatch, capsys, param_key, extreme_value, expected_max_return):
#     params_extreme = default_params.copy()
#     params_extreme[param_key] = extreme_value
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, params_extreme, symbol="AAPL", years="1")
#     assert isinstance(perf_factor, float), f"Pipeline with {param_key}={extreme_value} did not return a float."
#     assert perf_factor <= expected_max_return, f"Expected performance factor <= {expected_max_return} for {param_key}={extreme_value}, got {perf_factor}."
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("param_key, invalid_value", [
#     ("shortWindow", -10),
#     ("longWindow", 0),
#     ("RSIWindow", -5),
#     ("bollingerBands", -1),
# ])
# def test_pipeline_handles_invalid_parameter_values(monkeypatch, capsys, param_key, invalid_value):
#     params_invalid = default_params.copy()
#     params_invalid[param_key] = invalid_value
#     with pytest.raises(ValueError, match=f"Invalid value for {param_key}"):
#         run_pipeline_with_params(monkeypatch, capsys, params_invalid, symbol="AAPL", years="1")

# @pytest.mark.parametrize("symbol, years", [
#     ("INVALID_TICKER", "1"),
#     ("", "1"),
# ])
# def test_pipeline_handles_invalid_symbols(monkeypatch, capsys, symbol, years):
#     inputs = iter([symbol, years])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     output = capsys.readouterr().out
#     assert "Invalid symbol" in output, f"Expected invalid symbol error for {symbol}. Output: {output}"

# @pytest.mark.parametrize("years", ["0", "-1", "abc"])
# def test_pipeline_handles_invalid_years(monkeypatch, capsys, years):
#     inputs = iter(["AAPL", years])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     output = capsys.readouterr().out
#     assert "Input not an integer" in output or "Cannot accept negatives" in output, \
#         f"Expected invalid year error for {years}. Output: {output}"

# def test_pipeline_with_very_short_period(monkeypatch, capsys):
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, default_params, years="1")
#     assert isinstance(perf_factor, float)
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"
#     output = capsys.readouterr().out
#     match_days = re.search(r"Successfully fetched (\d+) days of data", output)
#     assert match_days, "Could not find number of days fetched."
#     days_fetched = int(match_days.group(1))
#     assert days_fetched > 200, f"Expected >200 trading days in 1 year for AAPL, got {days_fetched}"

# def test_pipeline_handles_empty_data(monkeypatch, capsys):
#     inputs = iter(["INVALID_TICKER", "1"])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     output = capsys.readouterr().out
#     assert "Failed to fetch data" in output, "Expected failure message for empty data."
#     assert "Trading Strategy Results:" not in output, "Expected no trading results for empty data."

# def test_pipeline_handles_large_tolerance(monkeypatch, capsys):
#     params = default_params.copy()
#     tolerance = 1000  # Unrealistically high tolerance
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1")
#     assert isinstance(perf_factor, float), "Expected numeric return value."
#     assert perf_factor < 1.5, f"Expected performance factor < 1.5 for high tolerance, got {perf_factor}."
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"

# @pytest.mark.parametrize("symbol, years, num_days", [
#     ("AAPL", "1", 200),
#     ("TSLA", "2", 400),
#     ("MSFT", "5", 1000),
# ])
# def test_forecast_accuracy(monkeypatch, capsys, symbol, years, num_days):
#     inputs = iter([symbol, years])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     data = Main.getData(symbol, *Main.getDates(int(years), 0))
#     forecasted_data = Main.Forecast.forecastPricesHighLowVolume(data, num_days, 12)
#     assert not forecasted_data.empty, "Forecasted data should not be empty."
#     assert len(forecasted_data) == num_days, f"Expected {num_days} forecasted days, got {len(forecasted_data)}."
#     assert all(forecasted_data.columns == ["High", "Low", "Close", "Volume"]), "Forecasted data columns mismatch."

# @pytest.mark.parametrize("indicator_function, params, expected_signal_range", [
#     (Main.Indicators.computeMovingAverages, {"shortWindow": 10, "longWindow": 20}, (-1, 1)),
#     (Main.Indicators.computeRSI, {"window": 14}, (-1, 1)),
#     (Main.Indicators.computeMACD, {}, (-1, 1)),
#     (Main.Indicators.computeBollingerBands, {"window": 20}, (-1, 1)),
#     (Main.Indicators.computeOBV, {}, (-1, 1)),
#     (Main.Indicators.computeATR, {"window": 14}, (-1, 1)),
#     (Main.Indicators.computeStochasticOscillator, {"window": 14}, (-1, 1)),
#     # Additional tests for edge cases
#     (Main.Indicators.computeMovingAverages, {"shortWindow": 5, "longWindow": 50}, (-1, 1)),
#     (Main.Indicators.computeRSI, {"window": 7}, (-1, 1)),
#     (Main.Indicators.computeMACD, {}, (-1, 1)),
#     (Main.Indicators.computeBollingerBands, {"window": 10}, (-1, 1)),
#     (Main.Indicators.computeOBV, {}, (-1, 1)),
#     (Main.Indicators.computeATR, {"window": 7}, (-1, 1)),
#     (Main.Indicators.computeStochasticOscillator, {"window": 7}, (-1, 1)),
# ])
# def test_indicator_signal_quality(indicator_function, params, expected_signal_range):
#     data = pd.DataFrame({
#         "Close": [100, 102, 101, 105, 107, 110],
#         "High": [101, 103, 102, 106, 108, 111],
#         "Low": [99, 100, 100, 104, 106, 109],
#         "Volume": [1000, 1200, 1100, 1300, 1400, 1500],
#     })
#     processed_data = indicator_function(data, **params)
#     signal_column = [col for col in processed_data.columns if col.startswith("Sig")][0]
#     signals = processed_data[signal_column]
#     assert signals.min() >= expected_signal_range[0], f"Signal values below expected range: {signals.min()}"
#     assert signals.max() <= expected_signal_range[1], f"Signal values above expected range: {signals.max()}"

# @pytest.mark.parametrize("params, expected_min_return, expected_max_return", [
#     ({"shortWindow": 5, "longWindow": 10, "RSIWindow": 7, "bollingerBands": 10, "ATR": 7, "stochasticOscillator": 7}, 0.0, 3.0),
#     ({"shortWindow": 50, "longWindow": 100, "RSIWindow": 14, "bollingerBands": 20, "ATR": 14, "stochasticOscillator": 14}, 0.0, 2.0),
# ])
# def test_pipeline_with_custom_parameters(monkeypatch, capsys, params, expected_min_return, expected_max_return):
#     perf_factor = run_pipeline_with_params(monkeypatch, capsys, params, symbol="AAPL", years="1")
#     assert isinstance(perf_factor, float), "Expected numeric return value."
#     assert expected_min_return <= perf_factor <= expected_max_return, f"Performance factor out of expected range: {perf_factor}"
#     assert profitability_ok(perf_factor), f"Profitability out of bounds: got {perf_factor}"