---
title: 'calzone: A Python package for measring calibration of probablistic models for classification'
tags:
  - Python
  - Machine Learning
  - Artificial Intelligence
  - Calibration
  - Probablistic models
authors:
  - name: Kwok Lung Fan
    orcid: 0000-0002-8246-4751
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Gene Pennello
    orcid: 0000-0002-9779-1165
    affiliation: 1
  - name: Qi Liu
    orcid: 0000-0002-4053-4213
    affiliation: 1
  - name: Nicholas Petrick
    orcid: 0000-0001-5167-8899
    affiliation: 1
  - name: Ravi K. Samala
    orcid: 0000-0002-6661-4801
    affiliation: 1
  - name: Frank W. Samuelson
    orcid: 0000-0002-1130-0303
    affiliation: 1
  - name: Yee Lam Elim Thompson
    orcid: 0000-0001-9653-7707
    affiliation: 1
  - name: Qian Cao
    affiliation: 1
    correspondence: "yes"
    email: qian.cao@fda.hhs.gov

affiliations:
 - name: U.S. Food and Drug Administration
   index: 1

date: 3 Oct 2024
bibliography: paper.bib 
---

# Summary
`calzone` is a Python package for measuring calibration of probabilistic models for classification problems. It provides a set of functions and classes for calibration visualization and calibration metrics computation given a representative dataset with the model's predictions and the true labels. The metrics provided in `calzone` include the following: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow statistic (HL), Integrated Calibration Index (ICI), Spiegelhalter's Z-statistics and Cox's calibration slope/intercept. Some metrics come with variations such as binning scheme and top-class or class-wise.

# Statement of need
Classification is one of the most fundamental and important tasks in machine learning. The performance of classification models is often evaluated by a proper scoring rule, such as the cross-entropy or mean square error. Examination of the distinguishing power (resolution), such as AUC or Se/Sp are also used to evaluate the model performance. However, the reliability or calibration performance of the model is often overlooked. 

`@Brocker_decompose` has shown that the proper scoring rule can be decomposed into the resolution and reliability. That means even if the model has high resolution (high AUC), it may not be a reliable or calibrated model. In many high-risk machine learning applications, such as medical diagnosis, the reliability of the model is of paramount importance. 

We refer to calibration as the agreement between the predicted probability and the true posterior probability of a class-of-interest, $P(D=1|\hat{p}=p) = p$. This is defined as moderate calibration by @Calster_weak_cal

In the `calzone` package, we provide a set of functions and classes for calibration visualization and metrics computation. Existing libraries such as `scikit-learn` are often not dedicated to calibration metrics computation and don't provide calibration metrics computation that are widely used in the statistical literature. Most libraries for calibration are focusing calibrated the model instead of measuring the level of calibration with various metrics. `calzone` is dedicated to calibration metrics computation and visualization.

# Functionality

## Reliability Diagram

Reliability Diagram is a graphical representation of the calibration of a classification model [@Brocker_reldia]. It groups the predicted probabilities into bins and plots the mean predicted probability against the empirical frequency in each bin. The reliability diagram can be used to assess the calibration of the model and to identify any systematic errors in the predictions. In addition, we add the option to plot with error bars to show the confidence interval of the empirical frequency in each bin. The error bars are calculated using Wilson's score interval [@wilson_interval].

```python
from calzone.utils import reliability_diagram
from calzone.vis import plot_reliability_diagram

wellcal_dataloader = data_loader(
    data_path="example_data/simulated_welldata.csv"
)

reliability, confindence, bin_edges, bin_counts = reliability_diagram(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    num_bins=15,
    class_to_plot=1
) 

plot_reliability_diagram(
    reliability,
    confindence,
    bin_counts,
    error_bar=True,
    title='Class 1 reliability diagram for well calibrated data'
)
```
![Reliability Diagram for well calibrated data](../docs/build/html/_images/notebooks_reliability_diagram_5_0.png)

## Calibration metrics

`calzone` provides functions to compute various calibration metrics. `calzone` also has a `CalibrationMetrics()` class which allows the user to compute the calibration metrics in a more convenient way. The following are the metrics that are currently supported in `calzone`: 

### Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
Expected Calibration Error (ECE), Maximum Calibration Error (MCE) and binning-based methods [@guo_calibration;@Naeini_ece] aim to measure the average deviation between predicted probability and true probability. We provide the option to use equal-width binning or equal-frequency binning, labeled as ECE-H and ECE-C respectively. Users can also choose to compute the metrics for the class-of-interest or the top-class. In the case of class-of-interest, the program will treat it as a 1-vs-rest classification problem. It can be computed in `calzone` as follows:
```python
from calzone.metrics import calculate_ece_mce

reliability, confindence, bin_edges, bin_counts = reliability_diagram(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    num_bins=10,
    class_to_plot=1,
    is_equal_freq=False
)

ece_h_classone, mce_h_classone = calculate_ece_mce(
    reliability,
    confindence,
    bin_counts=bin_counts
)
```


### Hosmer-Lemeshow statistic (HL)
Hosmer-Lemeshow statistic (HL) is a statistical test for the calibration of a probabilistic model. It is a chi-square based test that compares the observed and expected number of events in each bin. The null hypothesis is that the model is well calibrated. HL-test first bins data into predicted probability bins (equal-width $H$ or equal-count $C$) and the test statistic is calculated as:
$$
\text{HL} = \sum_{m=1}^{M} \frac{(O_{1,m}-E_{1,m})^2}{E_{1,m}(1-\frac{E_{1,m}}{N_m})}  \sim \chi^2_{M-2}
$$
where $E_{1,m}$ is the expected number of class-of-interest events in the $\text{m}^{th}$ bin, $O_{1,m}$ is the observed number of class-of-interest events in the $\text{m}^{th}$ bin, $N_m$ is the total number of observations in the $\text{m}^{th}$ bin, and $M$ is the number of bins. In `calzone`, the HL-test can be computed as follows:
```python
from calzone.metrics import hosmer_lemeshow_test

HL_H_ts, HL_H_p, df = hosmer_lemeshow_test(
    reliability,
    confindence,
    bin_count=bin_counts
)
```


### Cox's calibration slope/intercept
Cox's calibration slope/intercept is a non-parametric method for assessing the calibration of a probabilistic model [@Cox]. A new logistic regression model is fitted to the data, with the predicted odds ($\frac{p}{1-p}$) as the dependent variable and the true probability as the independent variable. The slope and intercept of the regression line are then used to assess the calibration of the model. A slope of 1 and intercept of 0 indicates perfect calibration. To test whether the model is calibrated, fix the slope to 1 and fit the intercept. If the intercept is significantly different from 0, the model is not calibrated. Then, fix the intercept to 0 and fit the slope. If the slope is significantly different from 1, the model is not calibrated.
 In `calzone`, Cox's calibration slope/intercept can be computed as follows:

```python
from calzone.metrics import cox_regression_analysis

cox_slope, cox_intercept, cox_slope_ci, cox_intercept_ci = cox_regression_analysis(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    class_to_calculate=1,
    print_results=True,
    fix_slope=True
)
```
The values of the slope and intercept give you a sense of the form of miscalibration. A slope greater than 1 indicates that the model is overconfident at high probabilities and underconfident at low probabilities, and vice versa. An intercept greater than 0 indicates that the model is overconfident in general, and vice versa. Notice that even if the slope is 1 and the intercept is 0, the model might not be calibrated, as Cox's calibration analysis fails to capture some types of miscalibration.

### Integrated calibration index (ICI)

The integrated calibration index (ICI) is very similar to Expected calibration error (ECE). It also tries to measure the average deviation between predicted probability and true probability. However, ICI does not use binning to estimate the true probability of a group of samples with similar predicted probability. Instead, ICI uses curve smoothing techniques to fit the regression curve and uses the regression result as the true probability [@ICI_austin]. The ICI is then calculated using the following formula:
$$
\text{ICI} = \frac{1}{n}\sum_{i=1}^{n} |f(p_i)-p_i|
$$
where $f$ is the fitting function and $p$ is the predicted probability. The curve fitting is usually done with loess regression. However, it is possible to use any curve fitting method to calculate the ICI. In `calzone`, we provide Cox's ICI and loess ICI support while the user can also use any curve fitting method to calculate the ICI using functions in `calzone`.
```python
from calzone.metrics import (
    cox_regression_analysis,
    lowess_regression_analysis,
    cal_ICI_cox
)

### calculating cox ICI
cox_ici = cal_ICI_cox(
    cox_slope,
    cox_intercept,
    wellcal_dataloader.probs,
    class_to_calculate=1
)

### calculating loess ICI
loess_ici, lowess_fit_p, lowess_fit_p_correct = lowess_regression_analysis(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    class_to_calculate=1,
    span=0.5,
    delta=0.001,
    it=0
)
```


Notice that flexible curve fitting methods such as loess regression are very sensitive to the choice of span and delta parameters. The user can visualize the fitting result to avoid overfitting or underfitting.

### Spiegelhalter's Z-test

Spiegelhalter's Z-test is a test of calibration proposed by Spiegelhalter in 1986 [@spiegelhalter_z]. It uses the fact that the Brier score can be decomposed into:
$$
B = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)^2 = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)(1-2p_i) + \frac{1}{N} \sum_{i=1}^N p_i(1-p_i)
$$
And the TS of Z test is defined as:
$$
Z = \frac{B - E(B)}{\sqrt{\text{Var}(B)}} = \frac{ \sum_{i=1}^N (x_i - p_i)(1-2p_i)}{\sum_{i=1}^N (1-2p_i)^2 p_i (1-p_i)}
$$
and it is asymptotically distributed as a standard normal distribution. In `calzone`, it can be calculated using
```python
from calzone.metrics import spiegelhalter_z_test

z, p_value = spiegelhalter_z_test(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    class_to_calculate=1
)
```


### Metrics class
`calzone` also provides a class called `CalibrationMetrics()` to calculate all the metrics mentioned above. The user can also use this class to calculate the metrics.

```python
from calzone.metrics import CalibrationMetrics

metrics = CalibrationMetrics(class_to_calculate=1)

CalibrationMetrics.calculate_metrics(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    metrics='all'
)
```
# Other features
## Bootstrapping

`calzone` also provides bootstrapping to calculate the confidence intervals of the metrics. The user can specify the number of bootstrap samples and the confidence level. 
```python
from calzone.metrics import CalibrationMetrics

metrics = CalibrationMetrics(class_to_calculate=1)

CalibrationMetrics.bootstrap(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    metrics='all',
    n_samples=1000
)
```
and it will return a structured numpy array.

## Subgroup analysis
`calzone` will perform subgroup analysis by default in the command line user interface. If the user input csv file contains a subgroup column, the program will compute metrics for the entire dataset and for each subgroup.

## Prevalence adjustment
`calzone` also provides prevalence adjustment to account for prevalence changes between training data and testing data. Since calibration is defined using posterior probability, a mere shift in the prevalence of the testing data will result in miscalibration. It can be fixed by searching for the optimal derived original prevalence such that the adjusted probability minimizes a proper scoring rule such as cross-entropy loss. The formula of prevalence adjusted probability is:
$$
P'(D=1|\hat{p}=p) = \frac{\eta'/(1-\eta')}{(1/p-1)(\eta/(1-\eta))} = p'
$$
where $\eta$ is the prevalence of the testing data, $\eta'$ is the prevalence of the training data, and $p$ is the predicted probability [@weijie_prevalence_adjustment;@prevalence_shift;@gu_likelihod_ratio]. We search for the optimal $\eta'$ that minimizes the cross-entropy loss.

## Multiclass extension
`calzone` also provides multiclass extension to calculate the metrics for multiclass classification. The user can specify the class to calculate the metrics using a 1-vs-rest approach and test the calibration of each class. Alternatively, the user can transform the data and make problem become a top-class calibration problem. The top-class calibration has a similar format to binary classification, but the class 0 probability is defined as 1 minus the probability of the class with the highest probability, and the class 1 probability is defined as the probability of the class with the highest probability. The labels are transformed into whether the predicted class equals the true class, 0 if not and 1 if yes. Notice that the interpretation of some metrics may change in the top-class transformation.

## Command line interface
`calzone` also provides a command line interface to calculate the metrics. The user can visualize the calibration curve, calculate the metrics and their confidence intervals using the command line interface. To use the command line interface, the user can run `python cal_metrics.py -h` to see the help message.

# Acknowledgements
The authors acknowledge the Research Participation Program at the Center for Devices and Radiological Health administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and the U.S. Food and Drug Administration (FDA). 

# Conflicts of interest
The authors declare no conflicts of interest.

# References