"""
Unit tests for calzone.vis module.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
from calzone import vis

def test_plot_reliability_diagram_runs():
    reliabilities = np.array([[0.8, 0.9, 1.0]])
    confidences = np.array([[0.7, 0.85, 0.95]])
    bin_counts = np.array([[10, 20, 30]])
    fig = vis.plot_reliability_diagram(reliabilities, confidences, bin_counts, return_fig=True)
    assert fig is not None
    print("test_plot_reliability_diagram_runs completed")

def test_plot_roc_curve_runs():
    fpr = np.array([0.0, 0.1, 0.2, 1.0])
    tpr = np.array([0.0, 0.5, 0.8, 1.0])
    roc_auc = 0.85
    fig = vis.plot_roc_curve(fpr, tpr, roc_auc, class_to_plot=1, return_fig=True)
    assert fig is not None
    print("test_plot_roc_curve_runs completed")

def test_plot_roc_curve_multiclass():
    fpr = [np.array([0.0, 0.1, 1.0]), np.array([0.0, 0.2, 1.0])]
    tpr = [np.array([0.0, 0.6, 1.0]), np.array([0.0, 0.7, 1.0])]
    roc_auc = [0.9, 0.8]
    fig = vis.plot_roc_curve(fpr, tpr, roc_auc, class_to_plot=None, return_fig=True)
    assert fig is not None
    print("test_plot_roc_curve_multiclass completed")

def run_vis_tests():
    """Run all visualization tests."""
    test_plot_reliability_diagram_runs()
    test_plot_roc_curve_runs()
    test_plot_roc_curve_multiclass()
    print("All visualization tests completed successfully")

if __name__ == "__main__":
    run_vis_tests()
