# Claude Code Project Context - Forecasting Archive

## Project Overview
- **Type**: Time series forecasting capstone project (demand forecasting)
- **Data**: Daily sales data from 1981-1984 (1460 records)
- **Target**: Predict t+1 or t+1,t+2 from time t
- **Current Status**: Baseline models implemented and tested

## Project Structure
```
forecasting_archive/
├── data/
│   └── for_test.csv          # Test dataset (Date, Sales columns)
├── models/
│   ├── baseline.py           # (existing, minimal)
│   └── baseline_test.py      # Baseline model implementations
├── performance.py            # Evaluation metrics module
├── main_test.py              # Simple test script for Jupyter conversion
├── main_test.ipynb           # Jupyter notebook version
├── README.md                 # Korean documentation for users
└── CLAUDE.md                # This context file
```

## Modules Implemented

### 1. Performance Evaluation (`performance.py`)
- **Main Class**: `TimeSeriesEvaluator`
- **Default Metric**: WMAPE
- **Available Metrics**: WMAPE, MAPE, MAE, RMSE, MASE
- **Key Features**:
  - Extensible metric registry
  - Handles various input formats (pandas, numpy, list)
  - Seasonal period support for MASE
  - Convenience function: `evaluate_forecast()`

### 2. Baseline Models (`models/baseline_test.py`)
- **Main Class**: `BaselineModels`
- **Three Baseline Models**:
  1. **Last Day**: Use t-1 value for all horizons
  2. **Last Week**: Use t-7, t-6 values (seasonal naive)
  3. **Last Year**: Use t-365, t-364 values (accounting for leap years)

- **Key Methods**:
  - `evaluate_all_baselines()`: Run all models and return WMAPE scores
  - `plot_single_prediction_point()`: Focused dot plot visualization
  - `plot_baseline_comparison()`: Bar chart comparison

### 3. Test Script (`main_test.py`)
- Simple, cell-ready code for Jupyter conversion
- Tests both horizon=1 and horizon=2
- Generates performance metrics and visualizations
- Clean output formatting

## Key Results
**Performance Ranking (WMAPE %)**:
```
Model        | t+1    | t+1,t+2
-------------|--------|--------
last_day     | 21.04% | 24.09%  <- BEST
last_week    | 28.36% | 28.37%
last_year    | 30.04% | 30.06%
```

**Conclusion**: `last_day` baseline is the strongest performer for both horizons.

## Visualization Features
- **Focused Timeline**: Shows only relevant 1-week context + prediction points
- **Dot Plot Style**: Clear distinction between actual vs predicted values
- **Reference Points**: Shows source values for each baseline
- **Last Year Annotation**: Separate text box (not cluttering main plot)
- **Multiple Markers**: Square (last day), Triangle (last week), Diamond (last year)

## Technical Notes
- Data preprocessing handles date conversion automatically
- Minimum 365-day history required for yearly baseline
- Error handling for edge cases (short data, zero values)
- Extensible design for adding new baseline models
- Performance module independent and reusable

## Next Steps (Potential)
1. Advanced models (ARIMA, Prophet, LSTM, etc.)
2. Feature engineering module
3. Cross-validation framework
4. Model ensemble system
5. Real-time prediction pipeline

## User Workflow
1. Load data with `BaselineModels(data)`
2. Run `evaluate_all_baselines(horizon=1)` 
3. Use `plot_single_prediction_point()` for visualization
4. Export results or move to Jupyter notebook

## Code Quality
- Type hints throughout
- Comprehensive docstrings
- Error handling and validation
- Modular, extensible architecture
- Clean separation of concerns