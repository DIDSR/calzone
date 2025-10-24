<p align="center">
    <img src="https://github.com/DIDSR/calzone/blob/main/logo.png" width="225">
<!--     <img src="./docs/source/logo_firecamp.png" width="300"> -->

</p>

# Evaluating the Calibration of Probabilistic Models
![Docs](https://readthedocs.org/projects/calzone-docs/badge/)
[![PyPI version](https://badge.fury.io/py/calzone-tool.svg)](https://badge.fury.io/py/calzone-tool)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08026/status.svg)](https://doi.org/10.21105/joss.08026)

`Calzone` is a comprehensive Python package for calculating and visualizing metrics for assessing the calibration of models with probabilistic output.

## Features

- Supports multiple calibration metrics, including Spiegelhalter's Z-test, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow (HL) test, Cox regression analysis, and Loess regression analysis.
- Provides tools for creating reliability diagrams and ROC curves.
- Offers equal-space and equal-frequency binning options.
- Provides bootstrapped confidence intervals for each calibration metric.
- Supports prevelance adjustment to account for prevalance differences between enriched data and population data.
- Extends metrics to multiclass classification problems with one-vs-rest or top-class calculations.

_To accurately assess the calibration of machine learning models, it is essential to have a **comprehensive and representative testing dataset** with sufficient coverage of the prediction space that is also **independent of the model development dataset** (for training, tuning, and calibration). The calibration metrics are not meaningful if the dataset is not representative of true intended population._

## Installation

You can install the package using pip:
```
pip install calzone-tool
```

## Usage

Using `Calzone` in Python:
```python
import numpy as np
from scipy.stats import beta
from calzone.metrics import CalibrationMetrics

# Generate simulated data with beta-binomial distribution.
class1_proba = beta.rvs(0.5, 0.5, size=1000)
class0_proba = 1 - class1_proba
X = np.concatenate(
    (class0_proba.reshape(-1, 1), class1_proba.reshape(-1, 1)), axis=1
)
Y = np.random.binomial(1, p=class1_proba)

# Calculate calibration metrics.
cal_metrics = CalibrationMetrics(class_to_calculate=1)
cal_metrics.calculate_metrics(Y, X, metrics='all')
```

Also, an experimental build of the graphical user interface can now be downloaded at [https://github.com/DIDSR/calzone/releases/tag/v0.0.1-alpha](https://github.com/DIDSR/calzone/releases/tag/v0.0.2-alpha).

Alternatively, you can run `cal_metrics` for a command line interface.


## Running the Tests

To run the full test suite, including validation against external packages, you need to install additional test dependencies listed in `tests/test_requirements.txt`:

```bash
pip install -r tests/test_requirements.txt
```

You can then run the main test scripts directly:

```bash
python tests/test_metrics.py
python tests/test_results.py
python tests/test_utils.py
python tests/test_vis.py
```

Or run all tests together using:

```bash
python tests/main.py
```

Some tests require optional external packages (e.g., `scikit-learn`, `mapie`, `relplot`, `pycaleva`). Tests that depend on these will be skipped if the packages are not installed.

## Documentation

For a detailed manual and API reference, please visit our [documentation page](https://calzone-docs.readthedocs.io/en/latest/index.html).

## Support
If you encounter any issues or have questions about the package, please [open an issue request](https://github.com/DIDSR/calzone/issues) or contact:
* [Kwok Lung (Jason) Fan](mailto:kwoklung.fan@fda.hhs.gov?subject=calzone)
* [Qian Cao](mailto:qian.cao@fda.hhs.gov?subject=calzone)

## Cite
If you found this project useful in your academic work, we appreciate citing it using the following BibTeX entry:

```bibtex
@article{Fan2025Calzone,
  title   = {Calzone: A Python package for measuring calibration of probabilistic models for classification},
  author  = {Fan, Kwok Lung and Pennello, Gene and Liu, Qi and Petrick, Nicholas and Samala, Ravi K. and Samuelson, Frank W. and Thompson, Yee Lam Elim and Cao, Qian},
  year    = {2025},
  journal = {Journal of Open Source Software},
  volume  = {10},
  number  = {114},
  pages   = {8026},
  doi     = {10.21105/joss.08026},
  url     = {https://doi.org/10.21105/joss.08026},
}
```

## Disclaimer regarding Regulatory Science Tools
The enclosed tool is part of the [Catalog of Regulatory Science Tools](https://cdrh-rst.fda.gov/), which provides a peer-reviewed resource for stakeholders to use where standards and qualified Medical Device Development Tools (MDDTs) do not yet exist. These tools do not replace FDA-recognized standards or MDDTs. This catalog collates a variety of regulatory science tools that the FDA's Center for Devices and Radiological Health's (CDRH) Office of Science and Engineering Labs (OSEL) developed. These tools use the most innovative science to support medical device development and patient access to safe and effective medical devices. If you are considering using a tool from this catalog in your marketing submissions, note that these tools have not been qualified as [Medical Device Development Tools](https://www.fda.gov/medical-devices/medical-device-development-tools-mddt) and the FDA has not evaluated the suitability of these tools within any specific context of use. You may [request feedback or meetings for medical device submissions](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program) as part of the Q-Submission Program.
For more information about the Catalog of Regulatory Science Tools, email [RST_CDRH@fda.hhs.gov](mailto:RST_CDRH@fda.hhs.gov).
