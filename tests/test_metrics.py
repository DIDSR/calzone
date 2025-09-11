"""
This module contains test functions for calibration metrics.

The main test function generates synthetic data and tests the CalibrationMetrics class.
"""

import numpy as np
from calzone.metrics import CalibrationMetrics
from scipy.stats import beta


def run_test_metrics():
    """
    Test function for calibration metrics.
    
    Generates synthetic binary classification data and tests the CalibrationMetrics class.
    The test includes:
    - Generating random binary labels using binomial distribution
    - Creating probability predictions using beta distribution
    - Testing the CalibrationMetrics class with the generated data
    Notice that the test is not exhaustive and doesn't test for correctness of the metrics.
    Returns:
        None
    
    Raises:
        AssertionError: If the results from CalibrationMetrics are not in dictionary format
    """
    # Generate sample data
    np.random.seed(123)
    n_samples = 1000
    y_proba = np.zeros((n_samples,2))
    class1_proba = beta.rvs(0.5, 0.5, size=n_samples)
    y_proba[:, 1] = class1_proba
    y_proba[:, 0] = 1 - class1_proba  # Ensure
    y_true = np.random.binomial(1, p=class1_proba)

    # Test CalibrationMetrics class
    metrics = CalibrationMetrics(class_to_calculate=1)
    results = metrics.calculate_metrics(y_true, y_proba, metrics='all')
    assert isinstance(results, dict)
    print("✓ CalibrationMetrics results are in dictionary format")
    
    ### Deterministic results given the random seed
    assert results['ECE-H'] < 0.05
    print(f"✓ ECE-H value is {results['ECE-H']:.4f} < 0.05")
    
    assert results['HL-H p-value'] > 0.05
    print(f"✓ HL-H p-value is {results['HL-H p-value']:.4f} > 0.05")

    # Additional border-case: Cox regression with fix_intercept=True
    from calzone.metrics import cox_regression_analysis
    coef, intercept, coef_ci, intercept_ci = cox_regression_analysis(
        y_true, y_proba, fix_intercept=True
    )
    assert isinstance(coef, float)
    assert isinstance(intercept, float)
    assert isinstance(coef_ci, (tuple, np.ndarray))
    assert isinstance(intercept_ci, (tuple, np.ndarray))
    print("✓ Cox regression finished when fix_intercept=True")

    # Additional border-case: Cox regression with fix_slope=True
    coef2, intercept2, coef_ci2, intercept_ci2 = cox_regression_analysis(
        y_true, y_proba, fix_slope=True
    )
    assert isinstance(coef2, float)
    assert isinstance(intercept2, float)    
    assert isinstance(coef_ci2, (tuple, np.ndarray))
    assert isinstance(intercept_ci2, (tuple, np.ndarray))
    print("✓ Cox regression finished when fix_slope=True")
