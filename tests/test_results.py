"""
This module contains validation tests for calibration metrics against external packages.

The tests validate the calzone package implementation against:
- scikit-learn's calibration_curve for reliability diagrams
- MAPIE's ECE and Spiegelhalter's Z test implementations

Based on validation.ipynb notebook.
"""

import numpy as np
import pytest
from calzone.utils import reliability_diagram, data_loader
from calzone.metrics import calculate_ece_mce, spiegelhalter_z_test, hosmer_lemeshow_test
import os


class TestValidationAgainstExternalPackages:
    """Test suite for validating calzone metrics against external packages."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data for all tests."""
        # Get the path to the example data relative to the test file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(test_dir, "..", "example_data", "simulated_welldata.csv")
        cls.wellcal_dataloader = data_loader(data_path=data_path)
    
    def test_reliability_diagram_against_sklearn(self):
        """Test reliability diagram implementation against scikit-learn."""
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        # scikit-learn implementation
        scikit_reliability_H, scikit_confidence_H = calibration_curve(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs[:, 1], 
            n_bins=15, 
            strategy='uniform', 
            pos_label=1
        )
        scikit_reliability_C, scikit_confidence_C = calibration_curve(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs[:, 1], 
            n_bins=15, 
            strategy='quantile', 
            pos_label=1
        )
        
        # calzone implementation
        calzone_reliability_H, calzone_confidence_H, bin_edge_H, bin_count_H = reliability_diagram(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            class_to_plot=1, 
            is_equal_freq=False
        )
        calzone_reliability_C, calzone_confidence_C, bin_edge_C, bin_count_C = reliability_diagram(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            class_to_plot=1, 
            is_equal_freq=True
        )
        
        # Test equal-width binning
        reliability_diff_H = np.abs(scikit_reliability_H - calzone_reliability_H)
        confidence_diff_H = np.abs(scikit_confidence_H - calzone_confidence_H)
        
        # Assert differences are very small (within numerical precision)
        assert np.all(reliability_diff_H < 1e-4), f"Reliability differences too large: {reliability_diff_H}"
        assert np.all(confidence_diff_H < 1e-4), f"Confidence differences too large: {confidence_diff_H}"
        
        # Test equal-count binning
        reliability_diff_C = np.abs(scikit_reliability_C - calzone_reliability_C)
        confidence_diff_C = np.abs(scikit_confidence_C - calzone_confidence_C)
        
        assert np.all(reliability_diff_C < 1e-4), f"Reliability differences too large: {reliability_diff_C}"
        assert np.all(confidence_diff_C < 1e-4), f"Confidence differences too large: {confidence_diff_C}"
    
    def test_ece_against_mapie(self):
        """Test Expected Calibration Error implementation against MAPIE."""
        try:
            from mapie.metrics.calibration import top_label_ece 
        except ImportError:
            try: # older version
                from mapie.metrics import top_label_ece
            except ImportError:
                pytest.skip("MAPIE not available")
        
        # Calculate reliability diagrams for top class
        calzone_reliability_topclass_H, calzone_confidence_topclass_H, bin_edge_topclass_H, bin_count_topclass_H = reliability_diagram(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            class_to_plot=None, 
            is_equal_freq=False
        )
        calzone_reliability_topclass_C, calzone_confidence_topclass_C, bin_edge_topclass_C, bin_count_topclass_C = reliability_diagram(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            class_to_plot=None, 
            is_equal_freq=True
        )
        
        # Compare MAPIE and calzone equal-width binning
        mapie_ece_h = top_label_ece(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            split_strategy='uniform'
        )
        calzone_ece_h = calculate_ece_mce(
            calzone_reliability_topclass_H, 
            calzone_confidence_topclass_H, 
            bin_count_topclass_H
        )[0]
        
        # Assert ECE values are very close
        assert abs(mapie_ece_h - calzone_ece_h) < 1e-4, f"ECE-H difference too large: {abs(mapie_ece_h - calzone_ece_h)}"
        
        # Compare MAPIE and calzone equal-count binning
        mapie_ece_c = top_label_ece(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            split_strategy='quantile'
        )
        calzone_ece_c = calculate_ece_mce(
            calzone_reliability_topclass_C, 
            calzone_confidence_topclass_C, 
            bin_count_topclass_C
        )[0]
        
        assert abs(mapie_ece_c - calzone_ece_c) < 1e-4, f"ECE-C difference too large: {abs(mapie_ece_c - calzone_ece_c)}"
    
    def test_spiegelhalter_z_against_mapie(self):
        """Test Spiegelhalter's Z test statistic against MAPIE."""
        try:
            from mapie.metrics.calibration import spiegelhalter_statistic
        except ImportError:
            try:
                from mapie.metrics import spiegelhalter_statistic
            except ImportError:
                pytest.skip("MAPIE not available")
        
        # Compare the Z statistics (not p-values as MAPIE uses one-sided test)
        mapie_z_stat = spiegelhalter_statistic(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs[:, 1]
        )
        calzone_z_stat = spiegelhalter_z_test(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs
        )[0]
        
        assert abs(mapie_z_stat - calzone_z_stat) < 1e-4, f"Z statistic difference too large: {abs(mapie_z_stat - calzone_z_stat)}"
    
    def test_hosmer_lemeshow_test_structure(self):
        """Test Hosmer-Lemeshow test returns expected structure."""
        # Calculate reliability diagram with equal-count binning
        calzone_reliability_C, calzone_confidence_C, bin_edge_C, bin_count_C = reliability_diagram(
            self.wellcal_dataloader.labels, 
            self.wellcal_dataloader.probs, 
            num_bins=15, 
            class_to_plot=1, 
            is_equal_freq=True
        )
        
        # Test Hosmer-Lemeshow test
        hl_result = hosmer_lemeshow_test(
            calzone_reliability_C, 
            calzone_confidence_C, 
            bin_count_C, 
            df=len(bin_count_C)
        )
        
        # Check that result has expected structure
        assert len(hl_result) == 3, "HL test should return 3 values: test statistic, p-value, df"
        assert isinstance(hl_result[0], (int, float)), "Test statistic should be numeric"
        assert isinstance(hl_result[1], (int, float)), "P-value should be numeric"
        assert isinstance(hl_result[2], (int, float)), "Degrees of freedom should be numeric"
        
        # Check that p-value is between 0 and 1
        assert 0 <= hl_result[1] <= 1, f"P-value should be between 0 and 1, got {hl_result[1]}"
        
        # Check that test statistic is non-negative
        assert hl_result[0] >= 0, f"Test statistic should be non-negative, got {hl_result[0]}"
        
        # Check that degrees of freedom matches expected value
        assert hl_result[2] == len(bin_count_C), f"Degrees of freedom should be {len(bin_count_C)}, got {hl_result[2]}"

    def test_reliability_diagram_against_relplot(self):
        """Test calzone reliability diagram against relplot's prepare_rel_diagram_binned."""
        try:
            import relplot as rp
        except ImportError:
            pytest.skip("relplot not available")

        # Calculate reliability diagram using calzone
        calzone_reliability_H, calzone_confidence_H, bin_edge_H, bin_count_H = reliability_diagram(
            self.wellcal_dataloader.labels,
            self.wellcal_dataloader.probs,
            num_bins=15,
            class_to_plot=1,
            is_equal_freq=False
        )

        # Calculate reliability diagram using relplot
        result = rp.prepare_rel_diagram_binned(
            self.wellcal_dataloader.probs[:, 1],
            self.wellcal_dataloader.labels,
            nbins=15
        )

        # Compare reliability values
        reliability_diff = np.abs(result['buckets'] - calzone_reliability_H)
        assert np.all(reliability_diff < 1e-4), f"Reliability differences too large: {reliability_diff}"

    def test_hosmer_lemeshow_and_spiegelhalterz_against_pycaleva(self):
        """Test HL-C score and SpiegelhalterZ p-value against pycaleva CalibrationEvaluator."""
        try:
            from pycaleva import CalibrationEvaluator
        except ImportError:
            pytest.skip("pycaleva not available")

        # Calculate metrics using calzone
        from calzone.metrics import CalibrationMetrics
        metrics = CalibrationMetrics(class_to_calculate=1)
        results = metrics.calculate_metrics(
            self.wellcal_dataloader.labels,
            self.wellcal_dataloader.probs,
            metrics='all'
        )

        # Calculate metrics using pycaleva
        ce = CalibrationEvaluator(
            self.wellcal_dataloader.labels.flatten(),
            self.wellcal_dataloader.probs[:, 1],
            outsample=True,
            n_groups=10
        )

        # Compare HL-C score
        hl_stat_diff = np.abs(ce.hosmerlemeshow(verbose=False).statistic - results['HL-C score'])
        assert hl_stat_diff < 1e-4, f"HL-C score difference too large: {hl_stat_diff}"

        # Compare SpiegelhalterZ p-value
        z_pvalue_diff = np.abs(ce.z_test().pvalue - results["SpiegelhalterZ p-value"])
        assert z_pvalue_diff < 1e-4, f"SpiegelhalterZ p-value difference too large: {z_pvalue_diff}"


def run_validation_tests():
    """Run all validation tests.
    
    This function can be called from main.py or other test runners.
    """
    test_instance = TestValidationAgainstExternalPackages()
    test_instance.setup_class()
    
    print("Running validation tests against external packages...")
    
    try:
        test_instance.test_reliability_diagram_against_sklearn()
        print("✓ Reliability diagram validation against scikit-learn passed")
    except ImportError:
        print("- Skipped reliability diagram test (scikit-learn not available)")
    except Exception as e:
        print(f"✗ Reliability diagram test failed: {e}")

    try:
        test_instance.test_ece_against_mapie()
        print("✓ ECE validation against MAPIE passed")
    except ImportError:
        print("- Skipped ECE test (MAPIE not available)")
    except Exception as e:
        print(f"✗ ECE test failed: {e}")

    try:
        test_instance.test_reliability_diagram_against_relplot()
        print("✓ Reliability diagram validation against relplot passed")
    except ImportError:
        print("- Skipped Reliability diagram test (relplot not available)")
    except Exception as e:
        print(f"✗ Reliability diagram test failed: {e}")

    try:
        test_instance.test_spiegelhalter_z_against_mapie()
        print("✓ Spiegelhalter Z test validation against MAPIE passed")
    except ImportError:
        print("- Skipped Spiegelhalter Z test (MAPIE not available)")
    except Exception as e:
        print(f"✗ Spiegelhalter Z test failed: {e}")

    try:
        test_instance.test_hosmer_lemeshow_and_spiegelhalterz_against_pycaleva()
        print("✓ Hosmer-Lemeshow and Spiegelhalter Z test validation against pycaleva passed")
    except Exception as e:
        print(f"✗ Hosmer-Lemeshow and Spiegelhalter Z test validation against pycaleva failed: {e}")

    try:
        test_instance.test_hosmer_lemeshow_test_structure()
        print("✓ Hosmer-Lemeshow test structure validation passed")
    except Exception as e:
        print(f"✗ Hosmer-Lemeshow test failed: {e}")
    
    print("Validation tests completed.")


if __name__ == "__main__":
    run_validation_tests()
