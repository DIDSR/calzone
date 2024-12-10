.. calzone documentation master file, created by
   sphinx-quickstart on Mon Sep 23 14:26:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to the documentation for calzone
========================================

The calzone package provides a suite of tools for assessing and improving the calibration of machine learning models, particularly for binary classification tasks. It offers calibration metrics and visualization tools for displaying reliability diagrams.  Please read the summary and guide section in this documentation first before using the package.

Key features of calzone include:

* Calculation of calibration metrics (ECE, MCE, Hosmer-Lemeshow test, spiegelhalter z test, etc.)
* Visualization functions for reliability diagrams
* Bootstrapping capabilities for confidence interval estimation
* Subgroup analysis for calibration metrics
* Command line interface scripts for batch processing
* Multi-class extension by using 1-vs-rest or top-class only

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
   notebooks/topclass.ipynb
   notebooks/GUI.ipynb
   notebooks/validation.ipynb
   modules
