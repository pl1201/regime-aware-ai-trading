"""
Complete Validation Pipeline for Trading Models

This module implements a comprehensive validation pipeline that combines
walk-forward validation, Monte Carlo simulation, stress testing, and paper trading.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings
import json
import os
from datetime import datetime
import joblib

# Import validation modules
from algo_trading.validation.walk_forward_validator import WalkForwardValidator
from algo_trading.validation.monte_carlo_simulator import MonteCarloSimulator
from algo_trading.validation.stress_tester import StressTester
from algo_trading.validation.paper_trading import PaperTradingSimulator

class ValidationPipeline:
    """
    Complete Validation Pipeline for Trading Models
    """

    def __init__(
        self,
        model,
        output_dir: str = "validation_results",
        n_walk_forward_splits: int = 5,
        n_monte_carlo_simulations: int = 500,
        paper_trading_days: int = 30
    ):
        """
        Initialize Validation Pipeline

        Args:
            model: Trading model to validate
            output_dir: Directory to save results
            n_walk_forward_splits: Number of walk-forward splits
            n_monte_carlo_simulations: Number of Monte Carlo simulations
            paper_trading_days: Number of days for paper trading
        """
        self.model = model
        self.output_dir = output_dir
        self.n_walk_forward_splits = n_walk_forward_splits
        self.n_monte_carlo_simulations = n_monte_carlo_simulations
        self.paper_trading_days = paper_trading_days

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize validators
        self.walk_forward_validator = WalkForwardValidator(
            model,
            n_splits=n_walk_forward_splits,
            min_train_size=0.5,
            test_size=0.2,
            retrain_frequency=2
        )

        self.monte_carlo_simulator = MonteCarloSimulator(
            model,
            n_simulations=n_monte_carlo_simulations,
            noise_levels=[0.01, 0.05, 0.1, 0.15],
            parameter_perturbations={
                'confidence_threshold': 0.1,
                'risk_per_trade': 0.2,
                'sl_atr_multiplier': 0.15,
            }
        )

        self.stress_tester = StressTester(model)

        print(f"Validation Pipeline Initialized")
        print(f"   Output Directory: {output_dir}")
        print(f"   Walk-Forward Splits: {n_walk_forward_splits}")
        print(f"   Monte Carlo Simulations: {n_monte_carlo_simulations}")
        print(f"   Paper Trading Days: {paper_trading_days}")

    def run_complete_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run complete validation pipeline

        Args:
            X: Feature matrix
            y: Target labels
            features_df: DataFrame with additional features
            historical_data: Historical price data for paper trading

        Returns:
            Complete validation results
        """
        print("Starting Complete Validation Pipeline...")
        start_time = datetime.now()
        results = {}

        try:
            # 1. Walk-Forward Validation
            print("\n1. Running Walk-Forward Validation...")
            wfv_results = self.walk_forward_validator.validate(X, y, features_df=features_df)
            results['walk_forward'] = wfv_results
            self._save_results(wfv_results, "walk_forward_results.json")
            print("   Walk-Forward Validation completed")

            # 2. Monte Carlo Simulation
            print("\n2. Running Monte Carlo Simulation...")
            mc_results = self.monte_carlo_simulator.simulate(X, y, features_df)
            results['monte_carlo'] = mc_results
            self._save_results(mc_results, "monte_carlo_results.json")
            print("   Monte Carlo Simulation completed")

            # 3. Stress Testing
            print("\n3. Running Stress Testing...")
            stress_results = self.stress_tester.test_all_conditions(X, y, features_df)
            results['stress_test'] = stress_results
            self._save_results(stress_results, "stress_test_results.json")
            print("   Stress Testing completed")

            # 4. Paper Trading (if historical data provided)
            if historical_data is not None:
                print("\n4. Running Paper Trading Simulation...")
                paper_results = self._run_paper_trading(historical_data)
                results['paper_trading'] = paper_results
                self._save_results(paper_results, "paper_trading_results.json")
                print("   Paper Trading Simulation completed")

            # Calculate overall validation score
            validation_score = self._calculate_validation_score(results)
            results['validation_score'] = validation_score

            # Save complete results
            self._save_results(results, "complete_validation_results.json")

            # Generate reports
            self._generate_reports(results)

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\nCOMPLETE VALIDATION PIPELINE FINISHED")
            print(f"{'='*50}")
            print(f"Start Time: {start_time}")
            print(f"End Time:   {end_time}")
            print(f"Duration:   {duration}")
            print(f"Validation Score: {validation_score:.4f}")
            print(f"{'='*50}")

            return results

        except Exception as e:
            warnings.warn(f"Validation pipeline failed: {e}")
            results['error'] = str(e)
            self._save_results(results, "validation_error.json")
            return results

    def _run_paper_trading(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run paper trading simulation

        Args:
            historical_data: Historical price data

        Returns:
            Paper trading results
        """
        try:
            # Create paper trading simulator
            simulator = PaperTradingSimulator(
                initial_balance=10000.0,
                max_positions=5,
                slippage=0.001,
                commission=0.001
            )

            # Run backtest
            simulator.run_backtest(self.model, historical_data)

            # Get results
            metrics = simulator.get_performance_metrics()
            report = simulator.get_detailed_report()

            return {
                'metrics': metrics,
                'report': report,
                'final_balance': simulator.balance,
                'total_pnl': metrics['total_pnl'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown']
            }

        except Exception as e:
            warnings.warn(f"Paper trading failed: {e}")
            return {
                'error': str(e),
                'metrics': {},
                'report': "Paper trading failed"
            }

    def _calculate_validation_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall validation score

        Args:
            results: Validation results

        Returns:
            Validation score (0-1)
        """
        try:
            score_components = []

            # Walk-forward validation score (40% weight)
            if 'walk_forward' in results:
                wfv_metrics = results['walk_forward']
                wfv_score = wfv_metrics.get('overall_accuracy', 0.0)
                score_components.append(wfv_score * 0.4)

            # Monte Carlo robustness score (30% weight)
            if 'monte_carlo' in results:
                mc_metrics = results['monte_carlo']['robustness_metrics']
                mc_score = mc_metrics.get('robustness_score', 0.0)
                score_components.append(mc_score * 0.3)

            # Stress test robustness score (20% weight)
            if 'stress_test' in results:
                stress_metrics = results['stress_test']['overall_metrics']
                stress_score = stress_metrics.get('robustness_score', 0.0)
                score_components.append(stress_score * 0.2)

            # Paper trading score (10% weight)
            if 'paper_trading' in results and 'metrics' in results['paper_trading']:
                pt_metrics = results['paper_trading']['metrics']
                pt_score = 0.0
                if 'total_return' in pt_metrics:
                    # Normalize return to 0-1 range (assuming 100% is excellent)
                    pt_score = min(pt_metrics['total_return'] / 1.0, 1.0)  # Cap at 100% return
                score_components.append(pt_score * 0.1)

            if score_components:
                return sum(score_components)
            else:
                return 0.0

        except Exception as e:
            warnings.warn(f"Could not calculate validation score: {e}")
            return 0.0

    def _save_results(self, results: Dict[str, Any], filename: str):
        """
        Save results to file

        Args:
            results: Results dictionary
            filename: Output filename
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"   Results saved to {filepath}")
        except Exception as e:
            warnings.warn(f"Could not save results to {filename}: {e}")

    def _generate_reports(self, results: Dict[str, Any]):
        """
        Generate detailed validation reports

        Args:
            results: Validation results
        """
        try:
            # Generate summary report
            summary_report = self._generate_summary_report(results)
            summary_path = os.path.join(self.output_dir, "validation_summary_report.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            print(f"   Summary report saved to {summary_path}")

            # Generate detailed reports for each validation type
            if 'walk_forward' in results:
                wfv_report = self.walk_forward_validator.get_detailed_report()
                wfv_path = os.path.join(self.output_dir, "walk_forward_detailed_report.txt")
                with open(wfv_path, 'w', encoding='utf-8') as f:
                    f.write(wfv_report)

            if 'monte_carlo' in results:
                mc_report = self.monte_carlo_simulator.get_detailed_report()
                mc_path = os.path.join(self.output_dir, "monte_carlo_detailed_report.txt")
                with open(mc_path, 'w', encoding='utf-8') as f:
                    f.write(mc_report)

            if 'stress_test' in results:
                stress_report = self.stress_tester.get_detailed_report()
                stress_path = os.path.join(self.output_dir, "stress_test_detailed_report.txt")
                with open(stress_path, 'w', encoding='utf-8') as f:
                    f.write(stress_report)

            if 'paper_trading' in results:
                pt_report = results['paper_trading'].get('report', '')
                if pt_report:
                    pt_path = os.path.join(self.output_dir, "paper_trading_detailed_report.txt")
                    with open(pt_path, 'w', encoding='utf-8') as f:
                        f.write(pt_report)

        except Exception as e:
            warnings.warn(f"Could not generate reports: {e}")

    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate summary validation report

        Args:
            results: Validation results

        Returns:
            Summary report string
        """
        report = []
        report.append("VALIDATION PIPELINE SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now()}")
        report.append("")

        # Overall validation score
        validation_score = results.get('validation_score', 0.0)
        report.append(f"OVERALL VALIDATION SCORE: {validation_score:.4f}")
        report.append("-" * 40)

        # Walk-Forward Validation Results
        if 'walk_forward' in results:
            wfv_results = results['walk_forward']
            report.append("WALK-FORWARD VALIDATION:")
            report.append(f"  Overall Accuracy:  {wfv_results.get('overall_accuracy', 0.0):.4f}")
            report.append(f"  Overall F1-Score:  {wfv_results.get('overall_f1_score', 0.0):.4f}")
            report.append(f"  Mean Period Acc:   {wfv_results.get('mean_period_accuracy', 0.0):.4f}")
            report.append("")

        # Monte Carlo Simulation Results
        if 'monte_carlo' in results:
            mc_results = results['monte_carlo']['robustness_metrics']
            report.append("MONTE CARLO SIMULATION:")
            report.append(f"  Mean Accuracy:     {mc_results.get('mean_accuracy', 0.0):.4f}")
            report.append(f"  Std Accuracy:      {mc_results.get('std_accuracy', 0.0):.4f}")
            report.append(f"  Robustness Score:   {mc_results.get('robustness_score', 0.0):.4f}")
            report.append(f"  Failure Rate:      {mc_results.get('failure_rate', 0.0):.2%}")
            report.append("")

        # Stress Testing Results
        if 'stress_test' in results:
            stress_results = results['stress_test']['overall_metrics']
            report.append("STRESS TESTING:")
            report.append(f"  Mean Accuracy:      {stress_results.get('mean_accuracy', 0.0):.4f}")
            report.append(f"  Std Accuracy:       {stress_results.get('std_accuracy', 0.0):.4f}")
            report.append(f"  Robustness Score:   {stress_results.get('robustness_score', 0.0):.4f}")
            report.append("")

        # Paper Trading Results
        if 'paper_trading' in results:
            pt_results = results['paper_trading']
            report.append("PAPER TRADING:")
            report.append(f"  Final Balance:      ${pt_results.get('final_balance', 0.0):,.2f}")
            report.append(f"  Total PnL:          ${pt_results.get('total_pnl', 0.0):,.2f}")
            report.append(f"  Win Rate:           {pt_results.get('win_rate', 0.0):.2%}")
            report.append(f"  Max Drawdown:       {pt_results.get('max_drawdown', 0.0):.2%}")

        return "\n".join(report)

    def plot_validation_results(self):
        """
        Plot all validation results
        """
        try:
            # Plot walk-forward results
            self.walk_forward_validator.plot_results()

            # Plot Monte Carlo results
            self.monte_carlo_simulator.plot_results()

            # Plot stress test results
            self.stress_tester.plot_results()

        except Exception as e:
            warnings.warn(f"Could not plot results: {e}")

    def get_validation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Get validation recommendations based on results

        Args:
            results: Validation results

        Returns:
            List of recommendations
        """
        recommendations = []

        try:
            validation_score = results.get('validation_score', 0.0)

            if validation_score >= 0.8:
                recommendations.append("Model is highly validated and ready for live trading")
            elif validation_score >= 0.6:
                recommendations.append("Model is moderately validated, consider additional testing")
            elif validation_score >= 0.4:
                recommendations.append("Model needs significant improvement before live trading")
            else:
                recommendations.append("Model is not validated and should not be used for live trading")

            # Specific recommendations based on individual tests
            if 'walk_forward' in results:
                wfv_accuracy = results['walk_forward'].get('overall_accuracy', 0.0)
                if wfv_accuracy < 0.55:
                    recommendations.append("Improve model accuracy through feature engineering or hyperparameter tuning")

            if 'monte_carlo' in results:
                mc_failure_rate = results['monte_carlo']['robustness_metrics'].get('failure_rate', 1.0)
                if mc_failure_rate > 0.3:
                    recommendations.append("Improve model robustness with better regularization and feature selection")

            if 'stress_test' in results:
                stress_min_accuracy = results['stress_test']['overall_metrics'].get('min_accuracy', 0.0)
                if stress_min_accuracy < 0.4:
                    recommendations.append("Improve model performance under extreme conditions")

        except Exception as e:
            warnings.warn(f"Could not generate recommendations: {e}")

        return recommendations


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
            self.confidence_threshold = 0.6

        def fit(self, X, y, regime_ids=None, features_df=None):
            self.is_fitted = True
            return self

        def predict(self, X, features_df=None):
            return np.random.choice([-1, 0, 1], len(X))

    model = DummyModel()

    # Create sample historical data
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
    prices = 40000 + np.cumsum(np.random.randn(1000) * 100)

    historical_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.random.rand(1000) * 200,
        'low': prices - np.random.rand(1000) * 200,
        'close': prices + np.random.randn(1000) * 50,
        'volume': np.random.rand(1000) * 1000
    })

    # Create validation pipeline
    pipeline = ValidationPipeline(
        model,
        output_dir="test_validation_results",
        n_walk_forward_splits=3,
        n_monte_carlo_simulations=100,
        paper_trading_days=7
    )

    # Run validation
    print("Running Complete Validation Pipeline test...")
    results = pipeline.run_complete_validation(X, y, historical_data=historical_data)

    print("\nValidation completed!")
    print(f"Validation score: {results.get('validation_score', 0.0):.4f}")

    # Show recommendations
    recommendations = pipeline.get_validation_recommendations(results)
    print("\nRECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  {rec}")