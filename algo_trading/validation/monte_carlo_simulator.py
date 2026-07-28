"""
Monte Carlo Simulation for Trading Strategy Robustness Testing

This module implements Monte Carlo simulations to test the robustness
of trading strategies under various market conditions and parameter variations.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import joblib

class MonteCarloSimulator:
    """
    Monte Carlo Simulator for trading strategy robustness testing
    """

    def __init__(
        self,
        model,
        n_simulations: int = 1000,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.15],
        parameter_perturbations: Dict[str, float] = None
    ):
        """
        Initialize Monte Carlo Simulator

        Args:
            model: Trading model to test
            n_simulations: Number of Monte Carlo simulations
            noise_levels: List of noise levels to test
            parameter_perturbations: Dictionary of parameter perturbation percentages
        """
        self.model = model
        self.n_simulations = n_simulations
        self.noise_levels = noise_levels
        self.parameter_perturbations = parameter_perturbations or {
            'confidence_threshold': 0.1,  # 10% perturbation
            'risk_per_trade': 0.2,        # 20% perturbation
            'sl_atr_multiplier': 0.15,    # 15% perturbation
        }
        self.results = []
        self.robustness_metrics = {}

    def simulate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run Monte Carlo simulations

        Args:
            X: Feature matrix
            y: Target labels
            features_df: DataFrame with additional features

        Returns:
            Simulation results dictionary
        """
        print("Starting Monte Carlo Simulations...")
        print(f"Total simulations: {self.n_simulations}")
        print(f"Noise levels: {self.noise_levels}")

        # Store original model state
        try:
            import pickle
            original_model = pickle.dumps(self.model) if hasattr(self.model, '__dict__') else None
        except Exception as e:
            warnings.warn(f"Could not serialize model for Monte Carlo simulation: {e}")
            original_model = None

        simulation_results = []
        accuracy_scores = []
        f1_scores = []

        # Run simulations
        for sim_idx in range(self.n_simulations):
            if sim_idx % 100 == 0:
                print(f"   Running simulation {sim_idx + 1}/{self.n_simulations}")

            try:
                # Perturb parameters
                perturbed_model = self._perturb_model()

                # Add noise to features
                noise_level = np.random.choice(self.noise_levels)
                X_noisy = self._add_noise(X, noise_level)

                # Add noise to labels (label noise)
                y_noisy = self._add_label_noise(y, noise_level)

                # Train on perturbed data
                regime_ids = (np.arange(len(X_noisy)) % perturbed_model.n_experts).astype(int)
                perturbed_model.fit(X_noisy, y_noisy, regime_ids, features_df)

                # Test on original data
                predictions = perturbed_model.predict(X, features_df)

                # Calculate metrics
                accuracy = np.mean(predictions == y) if len(predictions) == len(y) else 0.0
                f1 = self._calculate_f1_score(y, predictions) if len(predictions) == len(y) else 0.0

                # Store results
                result = {
                    'simulation': sim_idx + 1,
                    'noise_level': noise_level,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'perturbed_parameters': self._get_perturbed_parameters(perturbed_model)
                }

                simulation_results.append(result)
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)

                # Restore original model
                if original_model:
                    try:
                        import pickle
                        self.model = pickle.loads(original_model)
                    except Exception as e:
                        warnings.warn(f"Could not deserialize model for Monte Carlo simulation: {e}")
                        # If deserialization fails, we'll continue with the perturbed model
                        # This is safer than failing the entire simulation

            except Exception as e:
                warnings.warn(f"Simulation {sim_idx + 1} failed: {e}")
                # Continue with next simulation

        # Calculate robustness metrics
        self.results = simulation_results
        self.robustness_metrics = self._calculate_robustness_metrics(accuracy_scores, f1_scores)

        print(f"\nMONTE CARLO SIMULATION RESULTS")
        print(f"{'='*50}")
        print(f"Mean Accuracy:     {self.robustness_metrics['mean_accuracy']:.4f}")
        print(f"Std Accuracy:      {self.robustness_metrics['std_accuracy']:.4f}")
        print(f"Min Accuracy:      {self.robustness_metrics['min_accuracy']:.4f}")
        print(f"Max Accuracy:      {self.robustness_metrics['max_accuracy']:.4f}")
        print(f"Accuracy CI (95%): {self.robustness_metrics['accuracy_ci']}")
        print(f"Mean F1-Score:     {self.robustness_metrics['mean_f1']:.4f}")
        print(f"Robustness Score:  {self.robustness_metrics['robustness_score']:.4f}")
        print(f"Failure Rate:      {self.robustness_metrics['failure_rate']:.2%}")
        print(f"{'='*50}")

        return {
            'simulation_results': simulation_results,
            'robustness_metrics': self.robustness_metrics
        }

    def _perturb_model(self):
        """
        Create perturbed version of the model with modified parameters

        Returns:
            Perturbed model
        """
        # Create a copy of the model
        perturbed_model = joblib.loads(joblib.dumps(self.model)) if hasattr(self.model, '__dict__') else self.model

        # Perturb parameters
        if hasattr(perturbed_model, 'confidence_threshold'):
            original_threshold = getattr(perturbed_model, 'confidence_threshold', 0.6)
            perturbation = np.random.normal(0, self.parameter_perturbations['confidence_threshold'] * original_threshold)
            perturbed_model.confidence_threshold = np.clip(
                original_threshold + perturbation, 0.5, 0.9
            )

        if hasattr(perturbed_model, 'risk_manager') and perturbed_model.risk_manager:
            risk_manager = perturbed_model.risk_manager
            if hasattr(risk_manager, 'config'):
                config = risk_manager.config
                if hasattr(config, 'max_risk_per_trade'):
                    original_risk = config.max_risk_per_trade
                    perturbation = np.random.normal(0, self.parameter_perturbations['risk_per_trade'] * original_risk)
                    config.max_risk_per_trade = np.clip(
                        original_risk + perturbation, 0.005, 0.05
                    )

                if hasattr(config, 'sl_atr_multiplier'):
                    original_sl_mult = config.sl_atr_multiplier
                    perturbation = np.random.normal(0, self.parameter_perturbations['sl_atr_multiplier'] * original_sl_mult)
                    config.sl_atr_multiplier = np.clip(
                        original_sl_mult + perturbation, 1.0, 3.0
                    )

        return perturbed_model

    def _add_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add Gaussian noise to feature matrix

        Args:
            X: Feature matrix
            noise_level: Standard deviation of noise as fraction of feature std

        Returns:
            Noisy feature matrix
        """
        if noise_level <= 0:
            return X

        # Calculate noise standard deviation
        X_std = np.std(X, axis=0)
        noise_std = noise_level * X_std

        # Add noise
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise

        return X_noisy

    def _add_label_noise(self, y: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add noise to labels by randomly flipping some labels

        Args:
            y: True labels
            noise_level: Fraction of labels to flip

        Returns:
            Noisy labels
        """
        if noise_level <= 0:
            return y

        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * noise_level)

        # Randomly select indices to flip
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)

        # Flip labels (-1 to 0, 0 to 1, 1 to -1)
        for idx in flip_indices:
            if y_noisy[idx] == -1:
                y_noisy[idx] = 0
            elif y_noisy[idx] == 0:
                y_noisy[idx] = 1
            else:  # y_noisy[idx] == 1
                y_noisy[idx] = -1

        return y_noisy

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

    def _get_perturbed_parameters(self, model) -> Dict[str, Any]:
        """
        Get perturbed parameters for logging

        Args:
            model: Model with perturbed parameters

        Returns:
            Dictionary of perturbed parameters
        """
        params = {}
        if hasattr(model, 'confidence_threshold'):
            params['confidence_threshold'] = model.confidence_threshold
        if hasattr(model, 'risk_manager') and model.risk_manager:
            risk_manager = model.risk_manager
            if hasattr(risk_manager, 'config'):
                config = risk_manager.config
                if hasattr(config, 'max_risk_per_trade'):
                    params['max_risk_per_trade'] = config.max_risk_per_trade
                if hasattr(config, 'sl_atr_multiplier'):
                    params['sl_atr_multiplier'] = config.sl_atr_multiplier
        return params

    def _calculate_robustness_metrics(
        self,
        accuracy_scores: List[float],
        f1_scores: List[float]
    ) -> Dict:
        """
        Calculate robustness metrics from simulation results

        Args:
            accuracy_scores: List of accuracy scores
            f1_scores: List of F1 scores

        Returns:
            Robustness metrics dictionary
        """
        if not accuracy_scores:
            return {
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'accuracy_ci': (0.0, 0.0),
                'mean_f1': 0.0,
                'robustness_score': 0.0,
                'failure_rate': 1.0
            }

        # Basic statistics
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        min_accuracy = np.min(accuracy_scores)
        max_accuracy = np.max(accuracy_scores)

        # Confidence interval (95%)
        if len(accuracy_scores) > 1:
            ci_lower, ci_upper = np.percentile(accuracy_scores, [2.5, 97.5])
        else:
            ci_lower, ci_upper = mean_accuracy, mean_accuracy

        # Mean F1-score
        mean_f1 = np.mean(f1_scores) if f1_scores else 0.0

        # Robustness score (combination of mean accuracy and stability)
        # Higher mean and lower std = higher robustness
        robustness_score = mean_accuracy * (1.0 / (1.0 + std_accuracy)) if std_accuracy > 0 else mean_accuracy

        # Failure rate (simulations with very low accuracy)
        failure_rate = np.mean([acc < 0.5 for acc in accuracy_scores])

        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'accuracy_ci': (ci_lower, ci_upper),
            'mean_f1': mean_f1,
            'robustness_score': robustness_score,
            'failure_rate': failure_rate
        }

    def plot_results(self):
        """
        Plot Monte Carlo simulation results
        """
        if not self.results:
            print("No results to plot")
            return

        # Extract data
        accuracies = [r['accuracy'] for r in self.results]
        f1_scores = [r['f1_score'] for r in self.results]
        noise_levels = [r['noise_level'] for r in self.results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy distribution
        ax1.hist(accuracies, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Accuracy Distribution')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.axvline(self.robustness_metrics['mean_accuracy'], color='red', linestyle='--',
                   label=f"Mean: {self.robustness_metrics['mean_accuracy']:.3f}")
        ax1.legend()

        # F1-score distribution
        ax2.hist(f1_scores, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('F1-Score Distribution')
        ax2.set_xlabel('F1-Score')
        ax2.set_ylabel('Frequency')
        ax2.axvline(self.robustness_metrics['mean_f1'], color='red', linestyle='--',
                   label=f"Mean: {self.robustness_metrics['mean_f1']:.3f}")
        ax2.legend()

        # Accuracy vs Noise Level
        ax3.scatter(noise_levels, accuracies, alpha=0.6)
        ax3.set_title('Accuracy vs Noise Level')
        ax3.set_xlabel('Noise Level')
        ax3.set_ylabel('Accuracy')

        # F1-score vs Noise Level
        ax4.scatter(noise_levels, f1_scores, alpha=0.6, color='orange')
        ax4.set_title('F1-Score vs Noise Level')
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('F1-Score')

        plt.tight_layout()
        plt.show()

    def get_detailed_report(self) -> str:
        """
        Generate detailed Monte Carlo simulation report

        Returns:
            Formatted report string
        """
        if not self.robustness_metrics:
            return "No simulation results available"

        report = []
        report.append("MONTE CARLO SIMULATION DETAILED REPORT")
        report.append("=" * 60)
        report.append(f"Total Simulations: {self.n_simulations}")
        report.append(f"Noise Levels Tested: {self.noise_levels}")
        report.append("")
        report.append("ROBUSTNESS METRICS:")
        report.append("-" * 30)
        report.append(f"Mean Accuracy:     {self.robustness_metrics['mean_accuracy']:.4f}")
        report.append(f"Std Accuracy:      {self.robustness_metrics['std_accuracy']:.4f}")
        report.append(f"Min Accuracy:      {self.robustness_metrics['min_accuracy']:.4f}")
        report.append(f"Max Accuracy:      {self.robustness_metrics['max_accuracy']:.4f}")
        report.append(f"Accuracy CI (95%): {self.robustness_metrics['accuracy_ci'][0]:.4f} - {self.robustness_metrics['accuracy_ci'][1]:.4f}")
        report.append(f"Mean F1-Score:     {self.robustness_metrics['mean_f1']:.4f}")
        report.append(f"Robustness Score:  {self.robustness_metrics['robustness_score']:.4f}")
        report.append(f"Failure Rate:      {self.robustness_metrics['failure_rate']:.2%}")
        report.append("")
        report.append("PARAMETER PERTURBATIONS:")
        report.append("-" * 30)
        for param, perturbation in self.parameter_perturbations.items():
            report.append(f"{param}: ±{perturbation*100:.1f}%")

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
            self.confidence_threshold = 0.6

        def fit(self, X, y, regime_ids=None, features_df=None):
            self.is_fitted = True
            return self

        def predict(self, X, features_df=None):
            return np.random.choice([-1, 0, 1], len(X))

    model = DummyModel()

    # Create simulator
    simulator = MonteCarloSimulator(
        model,
        n_simulations=100,
        noise_levels=[0.01, 0.05, 0.1],
        parameter_perturbations={
            'confidence_threshold': 0.1,
            'risk_per_trade': 0.2,
        }
    )

    # Run simulation
    print("Running Monte Carlo Simulation test...")
    results = simulator.simulate(X, y)
    print("\nSimulation completed!")
    print(f"Robustness score: {simulator.robustness_metrics['robustness_score']:.4f}")