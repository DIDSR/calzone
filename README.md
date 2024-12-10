# calzone: a python package for measuring calibration in probabilistic models
![Docs](https://readthedocs.org/projects/calzone-docs/badge/)

`calzone` is a comprehensive Python package for calculating and visualizing various metrics to assess the calibration of probabilistic models.
To accurately assess the calibration of machine learning models, it is essential to have a comprehensive and reprensative dataset with sufficient coverage of the prediction space. The calibration metrics is not meaningful if the dataset is not representative of true intended population.


## Features

- Supports multiple calibration metrics including Spiegelhalter's Z-test, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow test, Cox regression analysis, and Loess regression analysis
- Provides tools for creating reliability diagrams and ROC curves
- Offers both equal-space and equal-frequency binning options
- Boostrapping for confidence intervals for each calibration metrics
- Prevelance adjustment to account for prevalance change between enriched data and population data.
- Multiclass extension by 1-vs-rest or top-class only

## Installation

`calzone` package require installation of `numpy`, `scipy`, `matplotlib` and `statsmodels`.

You can install calzone using pip:
```
pip install -e "git+https://github.com/DIDSR/calzone.git#egg=calzone"
```

Alternatively, you can clone the repository and install it locally:
```
git clone https://github.com/DIDSR/calzone.git
cd calzone
pip install .
```
## Usage

run `python cal_metrics.py -h` to see the help information and usage. To use the package in your Python code, please refer to the examples in the documentation pages. 

A GUI is available by running `python GUI_cal_metrics.py`. Support for the GUI is experiment and requires additional dependencies (i.e., `nicegui`).

## Documentation

For a detailed manual and API reference, please visit our [documentation page](https://calzone-docs.readthedocs.io/en/latest/index.html).

## Support
If you encounter any issues or have questions about the package, please [open an issue request](https://github.com/DIDSR/calzone/issues) or contact the authors:
* [Kwok Lung (Jason) Fan](mailto:kwoklung.fan@fda.hhs.gov?subject=calzone)
* [Qian Cao](mailto:qian.cao@fda.hhs.gov?subject=calzone)

## Disclaimer 
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
