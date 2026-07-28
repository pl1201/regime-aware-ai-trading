"""
Stress Testing Framework for Trading Models

This module implements comprehensive stress testing to evaluate
model performance under extreme market conditions.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class StressTester:
    """
    Stress Tester for trading models under extreme conditions
    """

    def __init__(self, model):
        """
        Initialize Stress Tester

        Args:
            model: Trading model to test
        """
        self.model = model
        self.stress_results = {}
        self.extreme_conditions = self._define_extreme_conditions()

    def _define_extreme_conditions(self) -> Dict[str, Dict]:
        """
        Define extreme market conditions for testing

        Returns:
            Dictionary of extreme conditions with parameters
        """
        return {
            'high_volatility': {
                'description': 'Extreme price volatility (2x normal)',
                'volatility_multiplier': 2.0,
                'trend_magnitude': 0.5,
                'noise_level': 0.3
            },
            'low_volatility': {
                'description': 'Very low price volatility (0.3x normal)',
                'volatility_multiplier': 0.3,
                'trend_magnitude': 2.0,
                'noise_level': 0.1
            },
            'trending_market': {
                'description': 'Strong trending market',
                'trend_magnitude': 3.0,
                'volatility_multiplier': 0.8,
                'noise_level': 0.2
            },
            'ranging_market': {
                'description': 'Sideways ranging market',
                'trend_magnitude': 0.1,
                'volatility_multiplier': 1.2,
                'noise_level': 0.4
            },
            'flash_crash': {
                'description': 'Sudden market crash (5% drop)',
                'trend_magnitude': -5.0,
                'volatility_multiplier': 5.0,
                'noise_level': 1.0,
                'duration': 5  # minutes
            },
            'flash_rally': {
                'description': 'Sudden market rally (5% gain)',
                'trend_magnitude': 5.0,
                'volatility_multiplier': 5.0,
                'noise_level': 1.0,
                'duration': 5  # minutes
            },
            'high_noise': {
                'description': 'High noise, no clear trend',
                'trend_magnitude': 0.0,
                'volatility_multiplier': 1.5,
                'noise_level': 2.0
            },
            'market_open': {
                'description': 'Market opening volatility',
                'trend_magnitude': 1.0,
                'volatility_multiplier': 3.0,
                'noise_level': 0.8
            },
            'market_close': {
                'description': 'Market closing volatility',
                'trend_magnitude': 0.5,
                'volatility_multiplier': 2.5,
                'noise_level': 0.6
            },
            'news_impact': {
                'description': 'News-driven market impact',
                'trend_magnitude': 2.0,
                'volatility_multiplier': 4.0,
                'noise_level': 1.5
            }
        }

    def test_all_conditions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Test model under all extreme conditions

        Args:
            X: Feature matrix
            y: Target labels
            features_df: DataFrame with additional features

        Returns:
            Stress test results dictionary
        """
        print("Starting Stress Testing...")
        print(f"Testing {len(self.extreme_conditions)} extreme conditions")

        results = {}
        overall_performance = []

        for condition_name, condition_params in self.extreme_conditions.items():
            print(f"\nTesting: {condition_name}")
            print(f"   {condition_params['description']}")

            try:
                # Generate stressed data
                X_stressed, y_stressed = self._generate_stressed_data(
                    X, y, condition_params
                )

                # Test model
                condition_result = self._test_condition(
                    X_stressed, y_stressed, features_df, condition_name
                )

                results[condition_name] = condition_result
                overall_performance.append(condition_result['accuracy'])

                print(f"   Accuracy: {condition_result['accuracy']:.4f}")
                print(f"   F1-Score: {condition_result['f1_score']:.4f}")

            except Exception as e:
                warnings.warn(f"Stress test for {condition_name} failed: {e}")
                results[condition_name] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'condition': condition_params,
                    'error': str(e)
                }

        # Calculate overall metrics
        mean_accuracy = np.mean(overall_performance) if overall_performance else 0.0
        std_accuracy = np.std(overall_performance) if overall_performance else 0.0
        min_accuracy = np.min(overall_performance) if overall_performance else 0.0
        max_accuracy = np.max(overall_performance) if overall_performance else 0.0

        self.stress_results = {
            'condition_results': results,
            'overall_metrics': {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'min_accuracy': min_accuracy,
                'max_accuracy': max_accuracy,
                'robustness_score': self._calculate_robustness_score(results)
            }
        }

        print(f"\nSTRESS TESTING RESULTS")
        print(f"{'='*50}")
        print(f"Mean Accuracy:    {mean_accuracy:.4f}")
        print(f"Std Accuracy:     {std_accuracy:.4f}")
        print(f"Min Accuracy:     {min_accuracy:.4f}")
        print(f"Max Accuracy:     {max_accuracy:.4f}")
        print(f"Robustness Score: {self.stress_results['overall_metrics']['robustness_score']:.4f}")
        print(f"{'='*50}")

        return self.stress_results

    def _generate_stressed_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        condition_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stressed data based on condition parameters

        Args:
            X: Original feature matrix
            y: Original labels
            condition_params: Condition parameters

        Returns:
            Tuple of (stressed_X, stressed_y)
        """
        # Create copy of data
        X_stressed = X.copy()
        y_stressed = y.copy()

        # Apply volatility multiplier
        vol_mult = condition_params.get('volatility_multiplier', 1.0)
        if vol_mult != 1.0:
            # Add volatility-based noise
            noise_std = np.std(X_stressed, axis=0) * (vol_mult - 1.0) * 0.5
            noise = np.random.normal(0, noise_std, X_stressed.shape)
            X_stressed = X_stressed + noise

        # Apply trend modifications
        trend_mult = condition_params.get('trend_magnitude', 1.0)
        if trend_mult != 1.0:
            # Modify features to simulate trend
            trend_component = np.linspace(0, trend_mult, X_stressed.shape[0])
            for i in range(min(5, X_stressed.shape[1])):  # Apply to first 5 features
                X_stressed[:, i] += trend_component * 0.1

        # Apply noise level
        noise_level = condition_params.get('noise_level', 0.0)
        if noise_level > 0:
            noise_std = np.std(X_stressed, axis=0) * noise_level
            noise = np.random.normal(0, noise_std, X_stressed.shape)
            X_stressed = X_stressed + noise

        # Apply label modifications for extreme conditions
        if 'flash' in condition_params.get('description', '').lower():
            # Modify some labels for flash events
            n_modify = min(50, len(y_stressed) // 10)
            modify_indices = np.random.choice(len(y_stressed), n_modify, replace=False)
            if 'crash' in condition_params.get('description', '').lower():
                # Change some labels to -1 (sell signals)
                y_stressed[modify_indices] = -1
            elif 'rally' in condition_params.get('description', '').lower():
                # Change some labels to 1 (buy signals)
                y_stressed[modify_indices] = 1

        return X_stressed, y_stressed

    def _test_condition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame],
        condition_name: str
    ) -> Dict:
        """
        Test model under specific condition

        Args:
            X: Feature matrix
            y: Target labels
            features_df: DataFrame with additional features
            condition_name: Name of condition being tested

        Returns:
            Condition test results
        """
        try:
            # Create regime IDs for training
            regime_ids = (np.arange(len(X)) % self.model.n_experts).astype(int)

            # Train model on stressed data
            self.model.fit(X, y, regime_ids, features_df)

            # Test on original data
            predictions = self.model.predict(X, features_df)

            # Calculate metrics
            accuracy = np.mean(predictions == y) if len(predictions) == len(y) else 0.0
            f1_score = self._calculate_f1_score(y, predictions) if len(predictions) == len(y) else 0.0

            return {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'condition': self.extreme_conditions[condition_name],
                'samples_tested': len(y)
            }

        except Exception as e:
            warnings.warn(f"Condition test failed: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'condition': self.extreme_conditions[condition_name],
                'error': str(e)
            }

    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate weighted F1-score

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            F1-score
        """
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except ImportError:
            # Manual F1 calculation
            classes = np.unique(np.concatenate([y_true, y_pred]))
            f1_scores = []

            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Weight by class frequency
                weight = np.sum(y_true == cls) / len(y_true)
                f1_scores.append(f1 * weight)

            return sum(f1_scores) if f1_scores else 0.0

    def _calculate_robustness_score(self, results: Dict) -> float:
        """
        Calculate overall robustness score

        Args:
            results: Stress test results

        Returns:
            Robustness score (0-1)
        """
        accuracies = []
        for condition_result in results.values():
            if 'accuracy' in condition_result and condition_result['accuracy'] > 0:
                accuracies.append(condition_result['accuracy'])

        if not accuracies:
            return 0.0

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Robustness score: higher mean, lower std = better
        robustness = mean_accuracy * (1.0 / (1.0 + std_accuracy))
        return np.clip(robustness, 0.0, 1.0)

    def plot_results(self):
        """
        Plot stress test results
        """
        if not self.stress_results:
            print("No stress test results to plot")
            return

        results = self.stress_results['condition_results']
        conditions = list(results.keys())
        accuracies = [results[cond]['accuracy'] for cond in conditions]
        f1_scores = [results[cond]['f1_score'] for cond in conditions]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Accuracy by condition
        bars1 = ax1.bar(range(len(conditions)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Stress Test - Accuracy by Condition')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(conditions)))
        ax1.set_xticklabels(conditions, rotation=45, ha='right')

        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{accuracies[i]:.3f}', ha='center', va='bottom')

        # F1-score by condition
        bars2 = ax2.bar(range(len(conditions)), f1_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Stress Test - F1-Score by Condition')
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticks(range(len(conditions)))
        ax2.set_xticklabels(conditions, rotation=45, ha='right')

        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1_scores[i]:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def get_detailed_report(self) -> str:
        """
        Generate detailed stress test report

        Returns:
            Formatted report string
        """
        if not self.stress_results:
            return "No stress test results available"

        results = self.stress_results['condition_results']
        overall_metrics = self.stress_results['overall_metrics']

        report = []
        report.append("STRESS TESTING DETAILED REPORT")
        report.append("=" * 60)
        report.append(f"Overall Robustness Score: {overall_metrics['robustness_score']:.4f}")
        report.append(f"Mean Accuracy:            {overall_metrics['mean_accuracy']:.4f}")
        report.append(f"Std Accuracy:             {overall_metrics['std_accuracy']:.4f}")
        report.append(f"Min Accuracy:             {overall_metrics['min_accuracy']:.4f}")
        report.append(f"Max Accuracy:             {overall_metrics['max_accuracy']:.4f}")
        report.append("")
        report.append("CONDITION-BY-CONDITION RESULTS:")
        report.append("-" * 50)
        report.append("Condition          | Accuracy | F1-Score | Description")
        report.append("-" * 50)

        for condition_name, result in results.items():
            description = result['condition']['description'][:20]  # Truncate description
            accuracy = result['accuracy']
            f1_score = result['f1_score']
            report.append(f"{condition_name:<18} | {accuracy:8.4f} | {f1_score:8.4f} | {description}")

        return "\n".join(report)

    def get_failure_analysis(self) -> str:
        """
        Analyze conditions where model performed poorly

        Returns:
            Failure analysis report
        """
        if not self.stress_results:
            return "No stress test results available"

        results = self.stress_results['condition_results']
        poor_conditions = []

        for condition_name, result in results.items():
            if result['accuracy'] < 0.55:  # Threshold for "poor" performance
                poor_conditions.append((condition_name, result['accuracy'], result['condition']['description']))

        if not poor_conditions:
            return "No significant failures detected in stress testing"

        report = []
        report.append("FAILURE ANALYSIS - Conditions with Poor Performance:")
        report.append("-" * 60)
        report.append("Condition          | Accuracy | Description")
        report.append("-" * 60)

        for condition_name, accuracy, description in sorted(poor_conditions, key=lambda x: x[1]):
            report.append(f"{condition_name:<18} | {accuracy:8.4f} | {description}")

        return "\n".join(report)


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], n_samples)

    # Create dummy model (replace with actual model)
    class DummyModel:
        def __init__(self):
            self.n_experts = 4
            self.is_fitted = False

        def fit(self, X, y, regime_ids=None, features_df=None):
            self.is_fitted = True
            return self

        def predict(self, X, features_df=None):
            return np.random.choice([-1, 0, 1], len(X))

    model = DummyModel()

    # Create stress tester
    stress_tester = StressTester(model)

    # Run stress test
    print("Running Stress Test...")
    results = stress_tester.test_all_conditions(X, y)
    print("\nStress testing completed!")
    print(f"Robustness score: {stress_tester.stress_results['overall_metrics']['robustness_score']:.4f}")