---
title: 'calzone: A Python package for measuring calibration of probabilistic models for classification'
tags:
  - Python
  - Machine Learning
  - Artificial Intelligence
  - Calibration
  - Probablistic models
  - Metric
  - Evaluation
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
    corresponding: true
    email: qian.cao@fda.hhs.gov

affiliations:
 - name: U.S. Food and Drug Administration
   index: 1

date: 3 Oct 2024
bibliography: paper.bib 
---

# Summary
`calzone` is a Python package for evaluating the calibration of probabilistic outputs of classifier models. It provides a set of functions for visualizing calibration and computation of calibration metrics given a representative dataset with the model's predictions and true class labels. The metrics provided in `calzone` include: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow (HL) statistic, Integrated Calibration Index (ICI), Spiegelhalter's Z-statistics and Cox's calibration slope/intercept. The package is designed with versatility in mind. For many of the metrics, users can adjust the binning scheme and toggle between top-class or class-wise calculations. 

# Statement of need
Classification is one of the most common applications in machine learning. Classification models are often evaluated by a proper scoring rule - a scoring function that assigns the best score when predicted probabilities match the true probabilities - such as cross-entropy or mean square error [@gneiting2007strictly]. Examination of the discrimination performance (resolution), such as AUC or Se/Sp are also used to evaluate the model performance. However, the reliability or calibration performance of the model is often overlooked.
@DIAMOND199285 had shown that the resolution performance of a model does not indicate the reliability of the model. @Brocker_decompose later has shown that any proper scoring rule can be decomposed into the resolution and reliability. Thus even if the model has high resolution (high AUC), it may not be a reliable or calibrated model. In many high-risk machine learning applications, such as medical diagnosis, the reliability of the model is of paramount importance.

We define calibration as the agreement between the predicted probability and the true posterior probability of a class-of-interest, $P(D=1|\hat{p}=p) = p$. This has been defined as moderate calibration by @Calster_weak_cal .

In the `calzone` package, we provide a set of functions and classes for calibration visualization and metrics computation. Existing libraries such as `scikit-learn` are often not dedicated to calibration metrics computation or don't provide calibration metrics computation that are widely used in the statistical literature. Other libraries such as `uncertainty-toolbox` are focused on implementing calibration methods and visualization instead of ways to evaluate calibration [@uncertaintyToolbox]. 

# Functionality

## Reliability Diagram

The reliability diagram (also referred to as a calibration plot) is a graphical representation of the calibration of a classification model [@Brocker_reldia;steyerberg2010assessing]. It groups the predicted probabilities into bins and plots the mean predicted probability against the empirical frequency in each bin. The reliability diagram can be used to assess the calibration of the model and to identify any systematic errors in the predictions. In addition, `calzone` gives the option to also plot the confidence interval of the empirical frequency in each bin. The confidence intervals are calculated using Wilson's score interval [@wilson_interval]. We provide example data in the `example_data` folder which are simulated using a beta-binomial distribution [@beta-binomial]. The predicted probabilities are sampled from a beta distribution and the true labels are assigned using a Bernoulli trial with the sampled probabilities. Users can generate simulated data using the `fake_binary_data_generator` class in the `utils` module.

```python
from calzone.utils import reliability_diagram
from calzone.vis import plot_reliability_diagram

wellcal_dataloader = data_loader(
    data_path="example_data/simulated_welldata.csv"
)

reliability, confidence, bin_edges, bin_counts = reliability_diagram(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    num_bins=15,
    class_to_plot=1
) 

plot_reliability_diagram(
    reliability,
    confidence,
    bin_counts,
    error_bar=True,
    title='Class 1 reliability diagram for well calibrated data'
)
```
![Reliability Diagram for well calibrated data](../docs/build/html/_images/notebooks_reliability_diagram_5_0.png)

## Calibration metrics

`calzone` provides functions to compute various calibration metrics. The `CalibrationMetrics()` class allows the user to compute the calibration metrics in a more convenient way. The following are metrics that are currently supported in `calzone`: 

### Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
Expected Calibration Error (ECE), Maximum Calibration Error (MCE) and other binning-based methods [@guo_calibration;@Naeini_ece] aim to measure the average absolute deviation between predicted probability and true probability. We provide the option to use equal-width binning or equal-count binning, labeled as ECE-H and ECE-C respectively. Users can also choose to compute the metrics for the class-of-interest or the top-class. In the case of class-of-interest, `calzone` will evaluate the calibration of a one-vs-rest classification problem. The following snippet demonstrates how these metrics are calculated in our package:

```python
from calzone.metrics import calculate_ece_mce

reliability, confidence, bin_edges, bin_counts = reliability_diagram(
    wellcal_dataloader.labels,
    wellcal_dataloader.probs,
    num_bins=10,
    class_to_plot=1,
    is_equal_freq=False
)

ece_h_classone, mce_h_classone = calculate_ece_mce(
    reliability,
    confidence,
    bin_counts=bin_counts
)
```


### Hosmer-Lemeshow statistic (HL)
The Hosmer-Lemeshow (HL) statistical test is for evaluating the calibration of a probabilistic model. It is a chi-square-based test that compares the observed and expected number of events in each bin. The null hypothesis is that the model is well calibrated. HL-test first bins data into predicted probability bins (equal-width $H$ or equal-count $C$) and the test statistic is calculated as:
$$
\text{HL} = \sum_{m=1}^{M} \frac{(O_{1,m}-E_{1,m})^2}{E_{1,m}(1-\frac{E_{1,m}}{N_m})}  \sim \chi^2_{M-2}
$$
where $E_{1,m}$ is the expected number of class-of-interest events in the $\text{m}^{th}$ bin, $O_{1,m}$ is the observed number of class-of-interest events in the $\text{m}^{th}$ bin, $N_m$ is the total number of observations in the $\text{m}^{th}$ bin, and $M$ is the number of bins. In `calzone`, the HL-test can be computed as follows:
```python
from calzone.metrics import hosmer_lemeshow_test

HL_H_ts, HL_H_p, df = hosmer_lemeshow_test(
    reliability,
    confidence,
    bin_count=bin_counts
)
```
When performing the HL test on validation sets that are not used in training, the degree of freedom of the HL test changes from $M-2$ to $M$ [@hosmer2013applied]. Intuitively, $\frac{(O_{1,m}-E_{1,m})^2}{E_{1,m}(1-\frac{E_{1,m}}{N_m})}$ is the difference squared divided by the variance of a binomial distribution and follows a chi-square distribution with 1 degree of freedom. Hence, the sum of $M$ chi-square distributions with 1 degree of freedom is a chi-square distribution with $M$ degrees of freedom if the data has no effect on the model. The increase in degree of freedom for validation samples has often been overlooked but it is crucial for the test to maintain the correct type 1 error rate. In `calzone`, users can specify the degree of freedom of the HL test by setting the `df` parameter.

### Cox's calibration slope/intercept
Cox's calibration slope/intercept is a regression analysis method for assessing the calibration of a probabilistic model [@Cox], which doesn't require binning. A new logistic regression model is fitted to the data, with the predicted odds ($\frac{p}{1-p}$) as the independent variable and the outcome as the dependent variable. The slope and intercept of the regression line are then used to assess the calibration of the model. A slope of 1 and intercept of 0 indicates perfect calibration. To test whether the model is calibrated, fix the slope to 1 and fit the intercept. If the intercept is significantly different from 0, the model is not calibrated. Then, fix the intercept to 0 and fit the slope. If the slope is significantly different from 1, the model is not calibrated. Alternatively, the slope and intercept can be fitted and tested simultaneously using bivariate distribution [@McCullagh:1989]. This feature is not provided in `calzone` but user can extract the covariance matrix by printing the result and perform the test manually.
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
The slope and intercept values indicate the type of miscalibration. A slope >1 shows overconfidence at high probabilities and underconfidence at low probabilities (and vice versa). In other word, a slope < 1 (> 1) indicates that the spread of the predictied risks is too large (small) relative to the true risks. A positive intercept indicates general overconfidence (and vice versa). However, even with ideal slope and intercept values, the model may still be miscalibrated due to non-linear effects that Cox's analysis cannot detect.

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

Notice that flexible curve fitting methods such as Loess regression are very sensitive to the choice of span and delta parameters. The user can visualize the fitting result to avoid overfitting or underfitting.

### Spiegelhalter's Z-test

Spiegelhalter's Z-test is a test of calibration proposed by Spiegelhalter in 1986 [@spiegelhalter_z]. It uses the fact that the Brier score can be decomposed into:
$$
B = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)^2 = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)(1-2p_i) + \frac{1}{N} \sum_{i=1}^N p_i(1-p_i)
$$
And the TS of Z test is defined as:
$$
Z = \frac{B - E(B)}{\sqrt{\text{Var}(B)}} = \frac{ \sum_{i=1}^N (x_i - p_i)(1-2p_i)}{\sum_{i=1}^N (1-2p_i)^2 p_i (1-p_i)}
$$
and it is asymptotically distributed as a standard normal distribution. In `calzone`, it can be calculated using:
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
## Confidence intervals

In addition to point estimates of calibration performance, `calzone` also provides bootstrapping to calculate the confidence intervals of the metrics. The user can specify the number of bootstrap samples and the confidence level. 
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
and a structured numpy array will be returned.

## Subgroup analysis
`calzone` will perform subgroup analysis by default in the command line user interface. If the user input CSV file contains a subgroup column, the program will compute metrics for the entire dataset and for each subgroup.

## Prevalence adjustment
`calzone` also provides prevalence adjustment to account for prevalence changes between training data and testing data. Since calibration is defined using posterior probability, a mere shift in the disease prevalence of the testing data will result in miscalibration. It can be fixed by searching for the optimal derived original prevalence such that the adjusted probability minimizes a proper scoring rule such as cross-entropy loss. The formula of prevalence adjusted probability is:
$$
P'(D=1|\hat{p}=p) = \frac{\eta'/(1-\eta')}{(1/p-1)(\eta/(1-\eta))} = p'
$$
where $\eta$ is the prevalence of the testing data, $\eta'$ is the prevalence of the training data, and $p$ is the predicted probability [@weijie_prevalence_adjustment;@prevalence_shift;@gu_likelihod_ratio;@Prevalence_HORSCH]. We search for the optimal $\eta'$ that minimizes the cross-entropy loss. The user can specify also specify $\eta'$ and adjust the probability output directly if that is available.

## Multiclass extension
`calzone` also provides multiclass extension to calculate the metrics for multiclass classification. The user can specify the class to calculate the metrics using a 1-vs-rest approach and test the calibration of each class. Alternatively, the user can transform the data and make the problem become a top-class calibration problem. The top-class calibration has a similar format to binary classification, but the class 0 probability is defined as 1 minus the probability of the class with the highest probability, and the class 1 probability is defined as the probability of the class with the highest probability. The labels are transformed into whether the predicted class equals the true class, 0 if not and 1 if yes. Notice that the interpretation of some metrics may change in the top-class transformation.

## Validation of methods
We compared the results calculated by `calzone` with external packages for some metrics to ensure the correctness of the implementation. For reliability diagram, we compared the result with the `sklearn.calibration.calibration_curve()` function in `scikit-learn` [@scikit]. For top-class ECE and SpiegelHalter's Z score, we compared the result with the `MAPIE` package [@taquet2022mapie]. For Hosmer-Lemeshow statistic, we compared the result with the `ResourceSelection` package in R language [@ResourceSelection]. Their results are consistent with ours. For other metrics such as ICI, no external package is available, so we compared the result with ECE as they are similar and we obtained reasonably similar results. We include the validation codes in our documentation.

## Command line interface
`calzone` also provides a command line interface to calculate the metrics. The user can visualize the calibration curve, calculate the metrics and their confidence intervals using the command line interface. To use the command line interface, the user can run `python cal_metrics.py -h` to see the help message.

# Acknowledgements
The authors acknowledge the Research Participation Program at the Center for Devices and Radiological Health administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and the U.S. Food and Drug Administration (FDA). 

# Conflicts of interest
The authors declare no conflicts of interest.

# References
