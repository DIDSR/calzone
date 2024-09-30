.. calzone documentation master file, created by
   sphinx-quickstart on Mon Sep 23 14:26:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to the documentation for calzone
========================================

calzone is a Python package for calculation of various calibration metrics. This work is credited to Kwok Lung (Jason) Fan and Qian Cao.

The calzone package provides a suite of tools for assessing and improving the calibration of machine learning models, particularly in binary classification tasks. It offers various calibration metrics, methods for generating and manipulating calibration data, and visualization tools for reliability diagrams. Whether you're working on model evaluation, uncertainty quantification, or improving the reliability of probabilistic predictions, calzone offers the utilities you need.

Key features of calzone include:

* Calculation of various calibration metrics (e.g., ECE, MCE, Hosmer-Lemeshow test, spiegelhalter z test, etc.)
* Visualization functions for reliability diagrams
* Bootstrapping capabilities for confidence interval estimation
* Subgroup analysis for calibration metrics
* Provides graphical user interface for easy calculation
* Provides command line interface scripts for batch processing

We hope you find calzone useful in your machine learning and data science projects!

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
   modules