import numpy as np
import pandas as pd
from typing import Union, List, Dict, Callable, Optional


class TimeSeriesEvaluator:
    """Time series forecasting performance evaluator"""
    
    def __init__(self):
        # Registry for metric functions
        self.metric_registry: Dict[str, Callable] = {
            'wmape': self._wmape,
            'mape': self._mape, 
            'mae': self._mae,
            'rmse': self._rmse,
            'mase': self._mase
        }
    
    def evaluate(self, 
                 actual: Union[pd.Series, np.ndarray, List],
                 predicted: Union[pd.Series, np.ndarray, List],
                 metrics: Optional[List[str]] = None,
                 default_metric: str = 'wmape',
                 seasonal_period: int = 1) -> Dict[str, float]:
        """
        Evaluate time series forecasting performance.
        
        Parameters:
        -----------
        actual : array-like
            Ground truth values
        predicted : array-like  
            Predicted values
        metrics : List[str], optional
            List of metrics to calculate. If None, calculate all metrics
        default_metric : str, default='wmape'
            Default metric to show first
        seasonal_period : int, default=1
            Seasonal period for MASE calculation
            
        Returns:
        --------
        Dict[str, float]
            Evaluation results for each metric
        """
        
        # Convert inputs to numpy arrays
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Check length consistency
        if len(actual) != len(predicted):
            raise ValueError(f"Length mismatch: actual({len(actual)}), predicted({len(predicted)})")
        
        # Set metrics to calculate
        if metrics is None:
            metrics = list(self.metric_registry.keys())
        
        # Check for invalid metrics
        invalid_metrics = [m for m in metrics if m not in self.metric_registry]
        if invalid_metrics:
            raise ValueError(f"Unsupported metrics: {invalid_metrics}")
        
        # Calculate each metric
        results = {}
        for metric in metrics:
            if metric == 'mase':
                results[metric] = self.metric_registry[metric](actual, predicted, seasonal_period)
            else:
                results[metric] = self.metric_registry[metric](actual, predicted)
        
        # Move default metric to front if present
        if default_metric in results:
            ordered_results = {default_metric: results[default_metric]}
            ordered_results.update({k: v for k, v in results.items() if k != default_metric})
            return ordered_results
        
        return results
    
    def _wmape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Weighted Mean Absolute Percentage Error"""
        return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    
    def _mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(actual - predicted))
    
    def _rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Square Error"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def _mase(self, actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 1) -> float:
        """Mean Absolute Scaled Error"""
        if len(actual) <= seasonal_period:
            raise ValueError(f"Data length ({len(actual)}) must be greater than seasonal period ({seasonal_period})")
        
        # Calculate MAE of seasonal naive forecast
        naive_forecast = actual[:-seasonal_period]
        naive_mae = np.mean(np.abs(actual[seasonal_period:] - naive_forecast))
        
        if naive_mae == 0:
            return np.inf if np.any(actual != predicted) else 0
        
        # Calculate MASE
        mae = self._mae(actual, predicted)
        return mae / naive_mae
    
    def add_metric(self, name: str, metric_func: Callable[[np.ndarray, np.ndarray], float]):
        """
        Add a new metric to the evaluator.
        
        Parameters:
        -----------
        name : str
            Metric name
        metric_func : Callable
            Metric function that takes (actual, predicted) and returns float
        """
        self.metric_registry[name] = metric_func
        print(f"Metric '{name}' has been added.")
    
    def list_metrics(self) -> List[str]:
        """Return list of available metrics."""
        return list(self.metric_registry.keys())


# Convenience functions
def evaluate_forecast(actual, predicted, metrics=None, default_metric='wmape', seasonal_period=1):
    """
    Convenience function for time series forecasting evaluation.
    
    Parameters:
    -----------
    actual : array-like
        Ground truth values
    predicted : array-like
        Predicted values  
    metrics : List[str], optional
        List of metrics to calculate
    default_metric : str, default='wmape'
        Default metric
    seasonal_period : int, default=1
        Seasonal period for MASE calculation
        
    Returns:
    --------
    Dict[str, float]
        Evaluation results
    """
    evaluator = TimeSeriesEvaluator()
    return evaluator.evaluate(actual, predicted, metrics, default_metric, seasonal_period)


if __name__ == "__main__":
    # Test example
    np.random.seed(42)
    actual = np.random.randn(100).cumsum()
    predicted = actual + np.random.randn(100) * 0.1
    
    evaluator = TimeSeriesEvaluator()
    results = evaluator.evaluate(actual, predicted)
    
    print("Time Series Forecasting Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")