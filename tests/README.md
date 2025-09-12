# Calzone Tests

This directory contains test files for the calzone package.

## Test Files

- `test_metrics.py`: Basic tests for calibration metrics using synthetic data
- `test_results.py`: Validation tests that compare calzone implementations against external packages (scikit-learn, MAPIE)
- `test_utils.py`: Testing utility functions such as dataloader and ROC curve making
- `test_vis.py`: Tests for visualization functions
- `main.py`: Main test runner that executes all tests

## Running Tests

### Basic Tests
To run the basic tests with only core dependencies:
```bash
python main.py
```

### Validation Tests (with external packages)
To run validation tests against external packages, you need to install additional dependencies:

```bash
# Install test dependencies
pip install -r test_requirements.txt

# Run all tests including validation
python main.py
```

### Using pytest
You can also run tests using pytest (if installed):
```bash
# Run all tests
pytest

# Run specific test file
pytest test_results.py

# Run with coverage (requires pytest-cov)
pytest --cov=calzone
```

## Test Dependencies

The validation tests require additional packages for cross-validation:
- `scikit-learn`: For reliability diagram validation
- `mapie`: For ECE and Spiegelhalter Z test validation
- `pandas`: For data handling
- `pycaleva`: For HL and Spiegelhalter test validation
- `relplot`: For advanced reliability diagram

These are optional dependencies and tests will be skipped if not available.
