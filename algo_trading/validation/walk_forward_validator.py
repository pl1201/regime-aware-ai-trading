"""
Walk-Forward Validation Framework for Trading Models

This module implements comprehensive walk-forward validation to test
model performance on out-of-sample data across different market regimes.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from datetime import datetime, timedelta
import warnings
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

class WalkForwardValidator:
    """
    Walk-Forward Validation for trading models
    """

    def __init__(
        self,
        model,
        n_splits: int = 5,
        min_train_size: float = 0.6,
        test_size: float = 0.2,
        retrain_frequency: int = 1  # Retrain every N periods
    ):
        """
        Initialize Walk-Forward Validator

        Args:
            model: Trading model to validate
            n_splits: Number of walk-forward splits
            min_train_size: Minimum training size ratio
            test_size: Test size ratio
            retrain_frequency: How often to retrain model (every N periods)
        """
        self.model = model
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.retrain_frequency = retrain_frequency
        self.results = []
        self.period_results = []

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[pd.Series] = None,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Perform walk-forward validation

        Args:
            X: Feature matrix
            y: Target labels
            timestamps: Timestamps for each sample
            features_df: DataFrame with additional features

        Returns:
            Validation results dictionary
        """
        n_samples = len(X)

        # Create time-based splits
        splits = self._create_time_splits(n_samples)

        all_predictions = []
        all_true_labels = []
        period_scores = []

        last_trained_model = None
        training_count = 0

        print("Starting Walk-Forward Validation...")
        print(f"Total periods: {len(splits)}")

        for period_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\nPeriod {period_idx + 1}/{len(splits)}")
            print(f"   Train samples: {len(train_idx)}")
            print(f"   Test samples: {len(test_idx)}")

            # Check if we need to retrain
            should_retrain = (
                last_trained_model is None or
                training_count % self.retrain_frequency == 0 or
                period_idx == 0
            )

            if should_retrain:
                print("   Retraining model...")
                # Extract training data
                X_train = X[train_idx]
                y_train = y[train_idx]

                # Create regime IDs for training (simple time-based split)
                regime_ids = (np.arange(len(X_train)) % self.model.n_experts).astype(int)

                # Train model
                try:
                    self.model.fit(X_train, y_train, regime_ids, features_df.iloc[train_idx] if features_df is not None else None)
                    last_trained_model = self.model
                    training_count += 1
                    print("   Training completed")
                except Exception as e:
                    warnings.warn(f"Training failed for period {period_idx + 1}: {e}")
                    if last_trained_model is not None:
                        self.model = last_trained_model
                    else:
                        # Fallback to simple fit
                        self.model.fit(X_train, y_train)
            else:
                print("   Using previously trained model")

            # Test model
            X_test = X[test_idx]
            y_test = y[test_idx]

            try:
                # Get predictions
                predictions = self.model.predict(X_test, features_df.iloc[test_idx] if features_df is not None else None)

                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

                # Store results
                period_result = {
                    'period': period_idx + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'retrained': should_retrain,
                    'predictions': predictions,
                    'true_labels': y_test
                }

                self.period_results.append(period_result)
                period_scores.append(accuracy)

                all_predictions.extend(predictions)
                all_true_labels.extend(y_test)

                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   F1-Score: {f1:.4f}")

            except Exception as e:
                warnings.warn(f"Testing failed for period {period_idx + 1}: {e}")
                # Add dummy results
                period_result = {
                    'period': period_idx + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'retrained': should_retrain,
                    'predictions': np.array([]),
                    'true_labels': np.array([])
                }
                self.period_results.append(period_result)
                period_scores.append(0.0)

        # Calculate overall metrics
        if all_predictions and all_true_labels:
            overall_accuracy = accuracy_score(all_true_labels, all_predictions)
            overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
            overall_precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
            overall_recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        else:
            overall_accuracy = 0.0
            overall_f1 = 0.0
            overall_precision = 0.0
            overall_recall = 0.0

        # Compile results
        results = {
            'overall_accuracy': overall_accuracy,
            'overall_f1_score': overall_f1,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'mean_period_accuracy': np.mean(period_scores) if period_scores else 0.0,
            'std_period_accuracy': np.std(period_scores) if period_scores else 0.0,
            'period_results': self.period_results,
            'total_periods': len(splits),
            'total_trainings': training_count,
            'retrain_frequency': self.retrain_frequency
        }

        self.results = results

        print(f"\nWALK-FORWARD VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy:  {overall_accuracy:.4f}")
        print(f"Overall F1-Score:  {overall_f1:.4f}")
        print(f"Mean Period Acc:   {results['mean_period_accuracy']:.4f} ± {results['std_period_accuracy']:.4f}")
        print(f"Total Trainings:   {training_count}")
        print(f"Retrain Frequency: {self.retrain_frequency}")
        print(f"{'='*50}")

        return results

    def _create_time_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-based splits for walk-forward validation

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        indices = np.arange(n_samples)
        splits = []

        # Calculate split sizes
        min_train_samples = int(n_samples * self.min_train_size)
        test_samples = int(n_samples * self.test_size)
        train_samples = n_samples - test_samples

        if train_samples < min_train_samples:
            # Adjust test size if needed
            test_samples = n_samples - min_train_samples
            train_samples = min_train_samples

        # Create walk-forward splits
        step_size = (n_samples - min_train_samples - test_samples) // (self.n_splits - 1) if self.n_splits > 1 else 0

        for i in range(self.n_splits):
            if i == 0:
                # First split - use minimum train size
                train_end = min_train_samples
            else:
                # Subsequent splits - move forward by step_size
                train_end = min_train_samples + (i * step_size)

            test_start = train_end
            test_end = min(test_start + test_samples, n_samples)

            if test_end > test_start:
                train_idx = indices[:train_end]
                test_idx = indices[test_start:test_end]
                splits.append((train_idx, test_idx))

        return splits

    def plot_results(self):
        """
        Plot validation results
        """
        try:
            import matplotlib.pyplot as plt

            if not self.period_results:
                print("No results to plot")
                return

            periods = [r['period'] for r in self.period_results]
            accuracies = [r['accuracy'] for r in self.period_results]
            f1_scores = [r['f1_score'] for r in self.period_results]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Accuracy over time
            ax1.plot(periods, accuracies, 'b-o', label='Accuracy')
            ax1.set_title('Walk-Forward Validation - Accuracy Over Time')
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # F1-score over time
            ax2.plot(periods, f1_scores, 'g-s', label='F1-Score')
            ax2.set_title('Walk-Forward Validation - F1-Score Over Time')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('F1-Score')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def get_detailed_report(self) -> str:
        """
        Generate detailed validation report

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available"

        report = []
        report.append("WALK-FORWARD VALIDATION DETAILED REPORT")
        report.append("=" * 60)
        report.append(f"Overall Accuracy:  {self.results['overall_accuracy']:.4f}")
        report.append(f"Overall F1-Score:  {self.results['overall_f1_score']:.4f}")
        report.append(f"Overall Precision: {self.results['overall_precision']:.4f}")
        report.append(f"Overall Recall:    {self.results['overall_recall']:.4f}")
        report.append(f"Mean Period Accuracy: {self.results['mean_period_accuracy']:.4f} ± {self.results['std_period_accuracy']:.4f}")
        report.append(f"Total Periods:     {self.results['total_periods']}")
        report.append(f"Total Trainings:   {self.results['total_trainings']}")
        report.append(f"Retrain Frequency: {self.results['retrain_frequency']}")
        report.append("")
        report.append("PERIOD-BY-PERIOD RESULTS:")
        report.append("-" * 40)
        report.append("Period | Trn Size | Tst Size | Accur | F1    | Retrained")
        report.append("-" * 40)

        for result in self.period_results:
            report.append(
                f"{result['period']:6d} | "
                f"{result['train_size']:8d} | "
                f"{result['test_size']:8d} | "
                f"{result['accuracy']:5.3f} | "
                f"{result['f1_score']:5.3f} | "
                f"{'Y' if result['retrained'] else 'N':9s}"
            )

        return "\n".join(report)


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
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

    # Create validator
    validator = WalkForwardValidator(
        model,
        n_splits=5,
        min_train_size=0.5,
        test_size=0.2,
        retrain_frequency=2
    )

    # Run validation
    print("Running Walk-Forward Validation test...")
    results = validator.validate(X, y)
    print("\nValidation completed!")
    print(f"Results: {results['overall_accuracy']:.4f} accuracy")