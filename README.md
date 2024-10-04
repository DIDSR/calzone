# calzone: a python package for measuring calibration in probabilistic models

calzone is a comprehensive Python package for calculating and visualizing various metrics to assess the calibration of probabilistic models.To accurately assess the calibration of machine learning models, it is essential to have a comprehensive and reprensative dataset with sufficient coverage of the prediction space. The calibration metrics is not meaningful if the dataset is not representative of true intended population.


## Features

- Supports multiple calibration metrics including Spiegelhalter's Z-test, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow test, Cox regression analysis, and Loess regression analysis
- Provides tools for creating reliability diagrams and ROC curves
- Offers both equal-space and equal-frequency binning options
- Boostrapping for confidence intervals for each calibration metrics
- Prevelance adjustment to account for prevalance change between enriched data and population data.

## Installation

You can install calzone using pip:
pip install -e "git+https://github.com/DIDSR/calzone.git"

## Usage

run `python cal_metrics.py -h` to see the help information and usage. To use the package in your Python code, please refer to the examples in the documentation pages.

## Documentation

For detailed documentation and API reference, please visit our [documentation page]().
