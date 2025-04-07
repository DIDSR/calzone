<p align="center">
    <img src="https://github.com/DIDSR/calzone/blob/main/logo.png" width="225">
<!--     <img src="./docs/source/logo_firecamp.png" width="300"> -->

</p>

# Evaluating the Calibration of Probabilistic Models
![Docs](https://readthedocs.org/projects/calzone-docs/badge/)
[![PyPI version](https://badge.fury.io/py/calzone-tool.svg)](https://badge.fury.io/py/calzone-tool)

`calzone` is a comprehensive Python package for calculating and visualizing metrics for assessing the calibration of models with probabilistic output.

## Features

- Supports multiple calibration metrics including Spiegelhalter's Z-test, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow (HL) test, Cox regression analysis, and Loess regression analysis.
- Provides tools for creating reliability diagrams and ROC curves.
- Offers equal-space and equal-frequency binning options.
- Provides bootstrapped confidence intervals for each calibration metric.
- Supports prevelance adjustment to account for prevalance differences between enriched data and population data.
- Extends metrics to multiclass classification problems with one-vs-rest or top-class calculations.

_To accurately assess the calibration of machine learning models, it is essential to have a comprehensive and representative dataset with sufficient coverage of the prediction space. The calibration metrics are not meaningful if the dataset is not representative of true intended population._

## Installation

You can install the package using pip:
```
pip install calzone-tool
```

## Usage

Run `python cal_metrics.py -h` to see the help information and usage. To use the package in your Python code, please refer to the examples in the documentation pages. 

A GUI is available by running `python calzoneGUI.py` under the gui directory. Support for the GUI is experiment and requires additional dependencies (i.e., `nicegui`). An experimental executable file (Calzone.exe) can be found under gui/dist.  

Alternatively, you can run `cal_metrics`  directly after installation for Command line interface.

User can also use `calzone` inside python kernal. Here is an example of metrics calculation with simulated data.
```python
import numpy as np
from scipy.stats import beta
from calzone.metrics import CalibrationMetrics
### Generate simulated data with beta-binomial distribution
class1_proba = beta.rvs(0.5, 0.5, size=1000)
class0_proba = 1 - class1_proba
X = np.concatenate(
    (class0_proba.reshape(-1, 1), class1_proba.reshape(-1, 1)), axis=1
)

Y = np.random.binomial(1, p=class1_proba)
cal_metrics = CalibrationMetrics(class_to_calculate=1)
cal_metrics.calculate_metrics(Y, X, metrics='all')
```

## Documentation

For a detailed manual and API reference, please visit our [documentation page](https://calzone-docs.readthedocs.io/en/latest/index.html).

## Support
If you encounter any issues or have questions about the package, please [open an issue request](https://github.com/DIDSR/calzone/issues) or contact:
* [Kwok Lung (Jason) Fan](mailto:kwoklung.fan@fda.hhs.gov?subject=calzone)
* [Qian Cao](mailto:qian.cao@fda.hhs.gov?subject=calzone)

## Disclaimer 
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
