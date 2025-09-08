import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import sys
import os

# Add parent directory to path to import performance module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from performance import evaluate_forecast


class BaselineModels:
    """Baseline forecasting models for time series prediction"""
    
    def __init__(self, data: pd.DataFrame, date_col: str = 'Date', target_col: str = 'Sales'):
        """
        Initialize with time series data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with date and target columns
        date_col : str
            Name of date column
        target_col : str
            Name of target column
        """
        self.data = data.copy()
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col
        self.target_col = target_col
        
    def prepare_forecast_data(self, horizon: int = 1) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare data for forecasting with given horizon.
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon (1 for t+1, 2 for t+1,t+2)
            
        Returns:
        --------
        Tuple[np.ndarray, List[int]]
            (actual_values, valid_indices) where valid_indices indicate 
            which time points have enough historical data for prediction
        """
        target_values = self.data[self.target_col].values
        n = len(target_values)
        
        # Need at least 365 days of history for yearly baseline
        min_history = 365
        valid_indices = list(range(min_history, n - horizon + 1))
        
        return target_values, valid_indices
    
    def last_day_baseline(self, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline 1: Use previous day's value for all future horizons.
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (actual_values, predicted_values)
        """
        target_values, valid_indices = self.prepare_forecast_data(horizon)
        
        actual = []
        predicted = []
        
        for i in valid_indices:
            # Actual values for horizon
            actual_horizon = target_values[i:i+horizon]
            
            # Predicted values: use t-1 for all horizons
            predicted_horizon = [target_values[i-1]] * horizon
            
            actual.extend(actual_horizon)
            predicted.extend(predicted_horizon)
        
        return np.array(actual), np.array(predicted)
    
    def last_week_baseline(self, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline 2: Use same day of previous week.
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (actual_values, predicted_values)
        """
        target_values, valid_indices = self.prepare_forecast_data(horizon)
        
        actual = []
        predicted = []
        
        for i in valid_indices:
            # Actual values for horizon
            actual_horizon = target_values[i:i+horizon]
            
            # Predicted values: use values from 7 days ago
            predicted_horizon = []
            for h in range(horizon):
                if i - 7 + h >= 0:
                    predicted_horizon.append(target_values[i - 7 + h])
                else:
                    # Fallback to last available value
                    predicted_horizon.append(target_values[i-1])
            
            actual.extend(actual_horizon)
            predicted.extend(predicted_horizon)
        
        return np.array(actual), np.array(predicted)
    
    def last_year_baseline(self, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline 3: Use same day of previous year (considering leap years).
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (actual_values, predicted_values)
        """
        target_values, valid_indices = self.prepare_forecast_data(horizon)
        
        actual = []
        predicted = []
        
        for i in valid_indices:
            # Actual values for horizon
            actual_horizon = target_values[i:i+horizon]
            
            # Predicted values: use values from ~365 days ago
            predicted_horizon = []
            for h in range(horizon):
                # Try 365 days ago first, then 364 for leap year adjustment
                year_ago_idx = i - 365 + h
                if year_ago_idx >= 0:
                    predicted_horizon.append(target_values[year_ago_idx])
                else:
                    # Try 364 days ago
                    year_ago_idx = i - 364 + h
                    if year_ago_idx >= 0:
                        predicted_horizon.append(target_values[year_ago_idx])
                    else:
                        # Fallback to last available value
                        predicted_horizon.append(target_values[i-1])
            
            actual.extend(actual_horizon)
            predicted.extend(predicted_horizon)
        
        return np.array(actual), np.array(predicted)
    
    def evaluate_all_baselines(self, horizon: int = 1) -> Dict[str, float]:
        """
        Evaluate all baseline models and return WMAPE scores.
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with model names and their WMAPE scores
        """
        results = {}
        
        # Model 1: Last day
        actual, predicted = self.last_day_baseline(horizon)
        wmape = evaluate_forecast(actual, predicted, metrics=['wmape'])['wmape']
        results['last_day'] = wmape
        
        # Model 2: Last week
        actual, predicted = self.last_week_baseline(horizon)
        wmape = evaluate_forecast(actual, predicted, metrics=['wmape'])['wmape']
        results['last_week'] = wmape
        
        # Model 3: Last year
        actual, predicted = self.last_year_baseline(horizon)
        wmape = evaluate_forecast(actual, predicted, metrics=['wmape'])['wmape']
        results['last_year'] = wmape
        
        return results
    
    def get_prediction_timeline(self, horizon: int = 1, 
                              test_period_days: int = 30) -> Dict[str, Dict]:
        """
        Get prediction timeline for visualization.
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon
        test_period_days : int
            Number of days to show in test period
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary with train/test data and predictions for each model
        """
        target_values = self.data[self.target_col].values
        dates = self.data[self.date_col].values
        n = len(target_values)
        
        # Define test period (last test_period_days)
        test_start_idx = n - test_period_days
        train_end_idx = test_start_idx
        
        # Prepare timeline data
        timeline_data = {
            'train': {
                'dates': dates[:train_end_idx],
                'values': target_values[:train_end_idx]
            },
            'test': {
                'dates': dates[test_start_idx:],
                'values': target_values[test_start_idx:]
            },
            'predictions': {}
        }
        
        # Generate predictions for each model
        models = ['last_day', 'last_week', 'last_year']
        model_methods = [self.last_day_baseline, self.last_week_baseline, self.last_year_baseline]
        
        for model_name, model_method in zip(models, model_methods):
            pred_dates = []
            pred_values = []
            
            # Generate predictions for test period
            for i in range(test_start_idx, n - horizon + 1):
                # Get prediction for this time point
                if model_name == 'last_day':
                    pred = [target_values[i-1]] * horizon
                elif model_name == 'last_week':
                    pred = []
                    for h in range(horizon):
                        if i - 7 + h >= 0:
                            pred.append(target_values[i - 7 + h])
                        else:
                            pred.append(target_values[i-1])
                elif model_name == 'last_year':
                    pred = []
                    for h in range(horizon):
                        year_ago_idx = i - 365 + h
                        if year_ago_idx >= 0:
                            pred.append(target_values[year_ago_idx])
                        else:
                            year_ago_idx = i - 364 + h
                            if year_ago_idx >= 0:
                                pred.append(target_values[year_ago_idx])
                            else:
                                pred.append(target_values[i-1])
                
                # Add predictions to timeline
                for h in range(horizon):
                    if i + h < n:
                        pred_dates.append(dates[i + h])
                        pred_values.append(pred[h])
            
            timeline_data['predictions'][model_name] = {
                'dates': pred_dates,
                'values': pred_values
            }
        
        return timeline_data
    
    def plot_single_prediction_point(self, horizon: int = 1, target_date_idx: int = None):
        """
        Plot a single prediction point with focused context (dot plot style).
        
        Parameters:
        -----------
        horizon : int
            Forecasting horizon  
        target_date_idx : int, optional
            Index of target date to predict. If None, use last available date.
        """
        target_values = self.data[self.target_col].values
        dates = self.data[self.date_col].values
        n = len(target_values)
        
        # Set target date (default to last available for prediction)
        if target_date_idx is None:
            target_date_idx = n - horizon
        
        # Get context: show 7 days before target
        context_start = max(0, target_date_idx - 7)
        context_end = target_date_idx
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot context (week before)
        context_dates = dates[context_start:context_end]
        context_values = target_values[context_start:context_end]
        ax.plot(context_dates, context_values, 'g-', label='Historical Context', linewidth=2, alpha=0.7)
        
        # Plot actual target values (what we want to predict)
        target_dates = dates[target_date_idx:target_date_idx + horizon]
        actual_target_values = target_values[target_date_idx:target_date_idx + horizon]
        ax.scatter(target_dates, actual_target_values, c='blue', s=100, 
                  label='Actual Target', zorder=5, edgecolors='black', linewidth=2)
        
        # Get predictions for this specific point
        predictions = {}
        
        # Last day prediction
        pred_last_day = [target_values[target_date_idx - 1]] * horizon
        predictions['Last Day'] = pred_last_day
        
        # Last week prediction  
        pred_last_week = []
        for h in range(horizon):
            if target_date_idx - 7 + h >= 0:
                pred_last_week.append(target_values[target_date_idx - 7 + h])
            else:
                pred_last_week.append(target_values[target_date_idx - 1])
        predictions['Last Week'] = pred_last_week
        
        # Last year prediction
        pred_last_year = []
        for h in range(horizon):
            year_ago_idx = target_date_idx - 365 + h
            if year_ago_idx >= 0:
                pred_last_year.append(target_values[year_ago_idx])
            else:
                year_ago_idx = target_date_idx - 364 + h  
                if year_ago_idx >= 0:
                    pred_last_year.append(target_values[year_ago_idx])
                else:
                    pred_last_year.append(target_values[target_date_idx - 1])
        predictions['Last Year'] = pred_last_year
        
        # Plot predictions
        colors = ['red', 'orange', 'purple']
        markers = ['s', '^', 'D']
        model_names = ['Last Day', 'Last Week', 'Last Year']
        
        for i, model_name in enumerate(model_names):
            ax.scatter(target_dates, predictions[model_name], 
                      c=colors[i], s=80, marker=markers[i],
                      label=f'Predicted ({model_name})', zorder=4, alpha=0.8)
        
        # Add reference points within the context window
        # Show last day reference
        last_day_date = dates[target_date_idx - 1]
        last_day_value = target_values[target_date_idx - 1]
        ax.scatter(last_day_date, last_day_value, c='red', s=50, marker='o', 
                  alpha=0.5, label='Last Day Ref')
        
        # Show last week reference if within context
        if target_date_idx - 7 >= context_start:
            last_week_date = dates[target_date_idx - 7]
            last_week_value = target_values[target_date_idx - 7]
            ax.scatter(last_week_date, last_week_value, c='orange', s=50, marker='o', 
                      alpha=0.5, label='Last Week Ref')
        
        # Add small shaded area for target period
        if horizon == 1:
            ax.axvspan(target_dates[0], target_dates[0], color='#808080', alpha=0.3, 
                      label='Target Date')
        else:
            ax.axvspan(target_dates[0], target_dates[-1], color='#808080', alpha=0.3, 
                      label='Target Period')
        
        # Set focused x-axis limits (only recent context + target)
        x_min = context_dates[0] if len(context_dates) > 0 else target_dates[0]
        x_max = target_dates[-1]
        # Add small padding (half day on each side)
        padding = pd.Timedelta(hours=12)
        ax.set_xlim(x_min - padding, x_max + padding)
        
        # Add annotation for last year reference (separate from main plot)
        if target_date_idx - 365 >= 0:
            last_year_value = target_values[target_date_idx - 365]
            last_year_date = pd.to_datetime(dates[target_date_idx - 365])
            last_year_date_str = last_year_date.strftime('%Y-%m-%d')
            
            # Add text annotation showing last year value
            ax.text(0.02, 0.98, f'Last Year Reference:\n{last_year_date_str}: {last_year_value:.1f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='purple', alpha=0.1),
                   fontsize=10)
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title(f'Baseline Prediction Example - Horizon t+{horizon}')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format dates
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_baseline_comparison(self, results: Dict[str, float], 
                               title: str = "Baseline Model Comparison",
                               metric: str = "WMAPE"):
        """
        Plot comparison of baseline models.
        
        Parameters:
        -----------
        results : Dict[str, float]
            Dictionary with model names and scores
        title : str
            Plot title
        metric : str
            Metric name for y-axis label
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        scores = list(results.values())
        
        bars = ax.bar(models, scores, width=0.6, alpha=0.8)
        
        # Add value labels on top of bars
        for i, (model, score) in enumerate(results.items()):
            ax.text(i, score + max(scores) * 0.01, f'{score:.2f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Baseline Models')
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(title)
        ax.set_ylim(0, max(scores) * 1.15)
        
        # Color bars differently
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def run_complete_baseline_test(self, horizons: List[int] = [1, 2], 
                                 plot: bool = True) -> Dict[int, Dict[str, float]]:
        """
        Run complete baseline test for multiple horizons.
        
        Parameters:
        -----------
        horizons : List[int]
            List of forecasting horizons to test
        plot : bool
            Whether to plot results
            
        Returns:
        --------
        Dict[int, Dict[str, float]]
            Results for each horizon
        """
        all_results = {}
        
        for horizon in horizons:
            print(f"\n=== Baseline Test Results (Horizon: t+{horizon}) ===")
            results = self.evaluate_all_baselines(horizon)
            all_results[horizon] = results
            
            # Print results
            for model, score in results.items():
                print(f"{model}: {score:.4f}% WMAPE")
            
            # Find best baseline
            best_model = min(results, key=results.get)
            print(f"Best baseline: {best_model} ({results[best_model]:.4f}% WMAPE)")
            
            # Plot if requested
            if plot:
                self.plot_baseline_comparison(
                    results, 
                    title=f"Baseline Model Comparison (Horizon: t+{horizon})",
                    metric="WMAPE"
                )
        
        return all_results


def load_test_data(filepath: str) -> pd.DataFrame:
    """
    Load test data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Example usage
    data = load_test_data('../data/for_test.csv')
    baseline_tester = BaselineModels(data)
    
    # Run complete test
    results = baseline_tester.run_complete_baseline_test(horizons=[1, 2])
    
    print("\n=== Summary ===")
    for horizon, res in results.items():
        best_model = min(res, key=res.get)
        print(f"Horizon t+{horizon}: Best baseline is {best_model} with {res[best_model]:.4f}% WMAPE")