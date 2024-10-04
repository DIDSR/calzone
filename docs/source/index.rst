.. calzone documentation master file, created by
   sphinx-quickstart on Mon Sep 23 14:26:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to the documentation for calzone
========================================

calzone is a Python package for calculation of various calibration metrics. This work is credited to Kwok Lung (Jason) Fan and Qian Cao.

The calzone package provides a suite of tools for assessing and improving the calibration of machine learning models, particularly in binary classification tasks. It offers various calibration metrics and visualization tools for reliability diagrams.  Please read the summary and guide first  before using the package.

Key features of calzone include:

* Calculation of various calibration metrics (e.g., ECE, MCE, Hosmer-Lemeshow test, spiegelhalter z test, etc.)
* Visualization functions for reliability diagrams
* Bootstrapping capabilities for confidence interval estimation
* Subgroup analysis for calibration metrics
* Provides command line interface scripts for batch processing

To accurately assess the calibration of machine learning models, it is essential to have a comprehensive and reprensative dataset with sufficient coverage of the prediction space. The calibration metrics is not meaningful if the dataset is not representative of true intended population.


We hope you find calzone useful in your machine learning projects!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   notebooks/quickstart.ipynb
   notebooks/metrics_summary.ipynb
   notebooks/reliability_diagram.ipynb
   notebooks/ece_mce.ipynb
   notebooks/hl_test.ipynb
   notebooks/cox.ipynb
   notebooks/ici.ipynb
   notebooks/spiegelhalter_z.ipynb
   notebooks/prevalence_adjustment.ipynb
   notebooks/subgroup.ipynb
   notebooks/GUI.ipynb
   modules