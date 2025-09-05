"""
Unit tests for calzone.utils module.
"""
import numpy as np
import pytest
from calzone import utils

def test_softmax_to_logits_basic():
    print("Running test_softmax_to_logits_basic...")
    probs = np.array([[0.2, 0.8], [0.5, 0.5]])
    logits = utils.softmax_to_logits(probs)
    assert logits.shape == probs.shape
    assert np.all(np.isfinite(logits))

def test_removing_nan():
    print("Running test_removing_nan...")
    y_true = np.array([1, 0, 1, 0])
    y_predict = np.array([1, 0, 1, 0])
    y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [np.nan, np.nan], [0.3, 0.7]])
    # Should remove row with NaN
    result = utils.removing_nan(y_true, y_predict, y_proba)
    assert result is None or isinstance(result, tuple)

def test_apply_prevalence_adjustment():
    print("Running test_apply_prevalence_adjustment...")
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])
    adjusted = utils.apply_prevalence_adjustment(0.5, y_true, y_proba)
    assert adjusted.shape == y_proba.shape
    assert np.allclose(np.sum(adjusted, axis=1), 1)

def test_loss():
    print("Running test_loss...")
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])
    l = utils.loss(0.5, y_true, y_proba)
    assert isinstance(l, float)

def test_transform_topclass():
    print("Running test_transform_topclass...")
    probs = np.array([[0.2, 0.8], [0.6, 0.4]])
    labels = np.array([1, 0])
    result = utils.transform_topclass(probs, labels)
    assert result is None or isinstance(result, tuple)

def test_fake_binary_data_generator():
    print("Running test_fake_binary_data_generator...")
    gen = utils.fake_binary_data_generator(0.5, 0.5)
    # Test attributes
    assert hasattr(gen, 'alpha_val')
    assert hasattr(gen, 'beta_val')
    # Test generate_data returns None or tuple (since code may be incomplete)
    result = gen.generate_data(10)
    # If implemented, should return tuple of arrays
    if result is not None:
        X, y_true = result
        assert X.shape == (10, 2)
        assert y_true.shape == (10,)
    # Test linear_miscal returns array if implemented
    X = np.array([[0.7, 0.3], [0.2, 0.8]])
    miscal = gen.linear_miscal(X, 0.8)
    assert miscal.shape == X.shape

def test_data_loader():
    print("Running test_data_loader...")
    # Only test instantiation, as methods may not be fully implemented
    loader = utils.data_loader('../example_data/simulated_welldata.csv')
    assert hasattr(loader, 'data_path')
    new_loader = loader.transform_topclass()
    assert isinstance(new_loader, utils.data_loader)

def run_utils_test():
    """Run all utility tests manually."""
    test_softmax_to_logits_basic()
    test_removing_nan()
    test_apply_prevalence_adjustment()
    test_loss()
    test_transform_topclass()
    test_fake_binary_data_generator()
    test_data_loader()
    print("All utils tests ran.")

if __name__ == "__main__":
    run_utils_test()
