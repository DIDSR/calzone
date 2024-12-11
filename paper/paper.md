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
    orcid: 0000-0002-3900-0433
    corresponding: true
    email: qian.cao@fda.hhs.gov

affiliations:
 - name: U.S. Food and Drug Administration
   index: 1

date: 3 Oct 2024
bibliography: paper.bib 
---

# Summary
`calzone` is a Python package for evaluating the calibration of probabilistic outputs of classifier models. It provides a set of functions for visualizing calibration and computing of calibration metrics given a representative dataset with the model's predictions and the true class labels. The metrics provided in `calzone` include: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer-Lemeshow (HL) statistic, Integrated Calibration Index (ICI), Spiegelhalter's Z-statistics and Cox's calibration slope/intercept. The package is designed with versatility in mind. For many of the metrics, users can adjust the binning scheme and toggle between top-class or class-wise calculations. 

# Statement of need
Classification is one of the most common applications in machine learning. Classification models may output predicted probabilities that an input observation belongs to a particular class, and those probabilities are often evaluated by a proper scoring rule - a scoring function that assigns the best score when predicted probabilities match the true probabilities - such as cross-entropy or mean square error [@gneiting2007strictly]. Examination of the discrimination performance (resolution), such as AUC or Se/Sp are also used to evaluate model performance. These metrics may be sufficient if the output of the model is not meant to be a calibrated probability.

@DIAMOND199285 showed that the resolution (i.e., high performance) of a model does not indicate the reliability/calibration (i.e., how well predicted probabilities match true probabilities) of the model. @Brocker_decompose later showed that any proper scoring rule can be decomposed into the resolution and reliability. Thus, even if the model has high resolution, it may not be a reliable model. In many high-risk machine learning applications, such as medical diagnosis and prognosis, reliability greatly aids in the interpretability of the model when deciding among multiple treatment options.

We define calibration as the agreement between the predicted probability and the true posterior probability of a class-of-interest, $P(D=1|\hat{p}=p) = p$. This has been defined as moderate calibration by @Calster_weak_cal and is also referred as the reliability of the model.

In the `calzone` package, we provide a set of functions and classes for visualizing calibration and evaluating calibration metrics given a representative dataset from the intended population. Existing libraries such as `scikit-learn` lacks calibration metrics that are widely used in the statistical literature. Other libraries such as `uncertainty-toolbox` are focused on implementing calibration methods and not calibration assessment. [@uncertaintyToolbox]. 

# Software description

## Input data
To evaluate the calibration of a model, users need a representative dataset from the intended population. The dataset should contain the true class labels and the model's predicted probabilities. In `calzone`, the dataset can be loaded from a CSV file using the `data_loader` function. The description of the input CSV file format can be found in the `calzone` documentation. Alternatively, users can pass the true class labels and the model's predicted probabilities as NumPy arrays to the `calzone` functions.

## Reliability Diagram

The reliability diagram (also referred to as the calibration plot) is a graphical representation of the calibration of a classification model [@Murphy_reliability;@Brocker_reldia]. It groups the predicted probabilities into bins and plots the mean predicted probability against the empirical frequency in each bin. The reliability diagram can be used to qualitatively  assess the calibration of the model and to identify any systematic errors in the predictions. In addition, `calzone` gives the option to also plot the confidence interval of the empirical frequency in each bin. The confidence intervals are calculated using the Wilson's score interval [@wilson_interval].

 We provide example data in the `example_data` folder which are simulated using a beta-binomial distribution [@beta-binomial]. More description can be found in the documentation. In the example code below, we will instead use the scikit-learn function to simulate a random 2-class dataset of correlated normally distributed data and fit a simple logistic model to illustrate. \autoref{fig:reldia} shows an example of the reliability diagram for class 1 with 15 equal-width bins for a well-calibrated dataset, where the x-axis is the mean predicted probability and the y-axis is the empirical frequency.
```python
from calzone.utils import reliability_diagram
from calzone.vis import plot_reliability_diagram

# Generate a random binary classification dataset
import sklearn.linear_model
import sklearn.datasets

features, labels = sklearn.datasets.make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=0.2,
    random_state=1
)
model = sklearn.linear_model.LogisticRegression()
model.fit(features, labels)
probs = model.predict_proba(features)

reliability, confidence, bin_edges, bin_counts = reliability_diagram(
    labels,
    probs,
    num_bins=15,
    class_to_plot=1
) 

plot_reliability_diagram(
    reliability,
    confidence,
    bin_counts,
    error_bar=True,
    title='Reliability diagram'
)
```
![Reliability Diagram for class 1 with simulated data. \label{fig:reldia}](reliability_diagram_calzonepaper.png)

## Calibration metrics

`calzone` provides functions to compute various calibration metrics. The `CalibrationMetrics()` class allows the user to compute the calibration metrics in a more convenient way. The following are metrics that are currently supported in `calzone`: 

### Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) [@guo_calibration;@Naeini_ece] aim to measure the average and maximum absolute deviation between the predicted probability and true probability. We provide the option to use equal-width binning or equal-count binning, labeled as ECE-H and ECE-C respectively. Users can also choose to compute the metrics for the class-of-interest or the top-class. Top-class mean only the calibration of the class with the highest probability is evaluted. In the case of class-of-interest, `calzone` will evaluate the calibration of a one-vs-rest classification problem. The following snippet demonstrates how these metrics are calculated in our package:

```python
from calzone.metrics import calculate_ece_mce

reliability, confidence, bin_edges, bin_counts = reliability_diagram(
    labels,
    probs,
    num_bins=10,
    class_to_plot=1,
    is_equal_freq=False
)
### Both ECE and MCE are calculated at the same time
ece_h_classone, mce_h_classone = calculate_ece_mce(
    reliability,
    confidence,
    bin_counts=bin_counts
)
```


### Hosmer-Lemeshow statistic (HL)
The Hosmer-Lemeshow (HL) statistical test [@hl_test] is for evaluating the calibration of a probabilistic model. It is a chi-square-based test that compares the observed and expected number of events in each bin. The null hypothesis is that the model is well calibrated. HL-test first bins data into predicted probability bins (equal-width $H$ or equal-count $C$) and the test statistic is calculated as:
$$
\text{HL} = \sum_{m=1}^{M} \frac{(O_{1,m}-E_{1,m})^2}{E_{1,m}\left(1-\frac{E_{1,m}}{N_m}\right)}  \sim \chi^2_{M-2}
$$
where $E_{1,m}$ is the expected number of class-of-interest events in the $\text{m}^{th}$ bin, $O_{1,m}$ is the observed number of class-of-interest events in the $\text{m}^{th}$ bin, $N_m$ is the total number of observations in the $\text{m}^{th}$ bin, and $M$ is the number of bins. In `calzone`, the HL-test can be computed as follows:
```python
from calzone.metrics import hosmer_lemeshow_test

HL_H_ts, HL_H_p, df = hosmer_lemeshow_test(
    reliability,
    confidence,
    bin_count = bin_counts,
    df = len(bin_counts) - 2,
)
```
When performing the HL test on validation sets that are not used in training, the degree of freedom of the HL test changes from $M-2$ to $M$ [@hosmer2013applied]. Intuitively, $\frac{(O_{1,m}-E_{1,m})^2}{E_{1,m}\left(1-\frac{E_{1,m}}{N_m}\right)}$ is the difference squared divided by the variance of a binomial distribution and follows a chi-square distribution with 1 degree of freedom. Hence, the sum of $M$ chi-square distributions with 1 degree of freedom is a chi-square distribution with $M$ degrees of freedom if the data has no effect on the model. The increase in degree of freedom for validation samples has often been overlooked but it is crucial for the test to maintain the correct type 1 error rate. In `calzone`, the default degree of freedoms is $M-2$ and users should specify the degree of freedom of the HL test by setting the `df` parameter.

### Cox's calibration slope/intercept
Cox's calibration slope/intercept is a regression analysis method for assessing the calibration of a probabilistic model [@Cox], which doesn't require binning. A logistic regression model is fit to the data, with the predicted odds ($\frac{p}{1-p}$) as the independent variable and the outcome as the dependent variable. The slope and intercept of the regression line are then used to assess the calibration of the model. A slope of 1 and intercept of 0 indicates perfect calibration. To test whether the model is calibrated, fix the slope to 1 and fit the intercept. If the intercept is significantly different from 0, the model is not calibrated. Then, fix the intercept to 0 and fit the slope. If the slope is significantly different from 1, the model is not calibrated. Alternatively, the slope and intercept can be fitted and tested simultaneously using a bivariate distribution [@McCullagh:1989]. This feature is not provided in `calzone` but user can extract the covariance matrix by printing the result and perform the test manually.
In `calzone`, Cox's calibration slope/intercept can be computed as follows:

```python
from calzone.metrics import cox_regression_analysis

cox_slope, cox_intercept, cox_slope_ci, cox_intercept_ci = cox_regression_analysis(
    labels,
    probs,
    class_to_calculate=1,
    print_results=True,
    fix_slope=True
)
```
The slope and intercept values indicate the type of miscalibration. A slope >1 shows overconfidence at high probabilities and underconfidence at low probabilities (and vice versa). In other word, a slope < 1 (> 1) indicates that the spread of the predictied risks is too large (small) relative to the true risks. A positive intercept indicates general overconfidence (and vice versa). However, even with ideal slope and intercept values, the model may still be miscalibrated due to non-linear effects that Cox's analysis cannot detect.

### Integrated calibration index (ICI)

The integrated calibration index (ICI) is very similar to Expected calibration error (ECE). It also tries to measure the average deviation between the predicted probability and true probability. However, ICI does not use binning to estimate the true probability of a group of samples with similar predicted probability. Instead, ICI uses curve smoothing techniques to fit a regression curve and uses the regression result as the true probability [@ICI_austin]. The ICI is then calculated using the following formula:
$$
\text{ICI} = \frac{1}{n}\sum_{i=1}^{n} |f(p_i)-p_i|
$$
where $f$ is the fitting function and $p$ is the predicted probability. The curve fitting is usually done with Locally Weighted Scatterplot Smoothing (LOWESS). However, it is possible to use any curve fitting method to calculate the ICI. One possible altenatively is to use the Cox calibration result and calculate the average difference between the predicted probability and the estimated true probability from the curve. In `calzone`, we provide the Cox ICI and LOWESS ICI support while the user can also use any curve fitting method to calculate the ICI using functions in `calzone`.
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
    probs,
    class_to_calculate=1
)

### calculating LOWESS ICI
lowess_ici, lowess_fit_p, lowess_fit_p_correct = lowess_regression_analysis(
    labels,
    probs,
    class_to_calculate=1,
    span=0.5,
    delta=0.001,
    it=0
)
```

Notice that flexible curve fitting methods such as LOWESS regression are very sensitive to the choice of span and delta parameters. The user can visualize the fitting result to avoid overfitting or underfitting.

### Spiegelhalter's Z-test

Spiegelhalter's Z-test is a test of calibration proposed by Spiegelhalter in 1986 [@spiegelhalter_z]. It uses the fact that the Brier score can be decomposed into:
$$
B = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)^2 = \frac{1}{N} \sum_{i=1}^N (x_i - p_i)(1-2p_i) + \frac{1}{N} \sum_{i=1}^N p_i(1-p_i)
$$
And the test statistic (TS) of Z test is defined as:
$$
Z = \frac{B - E(B)}{\sqrt{\text{Var}(B)}} = \frac{ \sum_{i=1}^N (x_i - p_i)(1-2p_i)}{\sum_{i=1}^N (1-2p_i)^2 p_i (1-p_i)}
$$
and it is asymptotically distributed as a standard normal distribution. In `calzone`, it can be calculated using:
```python
from calzone.metrics import spiegelhalter_z_test

z, p_value = spiegelhalter_z_test(
    labels,
    probs,
    class_to_calculate=1
)
```

### Metrics class
`calzone` also provides a class called `CalibrationMetrics()` to calculate all the metrics mentioned above. The user can also use this class to calculate a list of metrics or all the metrics within a single function call. The function will return a dictionary containing the metrics name and their values. The metrics can be specified as a list of strings. The string 'all' can be used to calculate all the metrics.

```python
from calzone.metrics import CalibrationMetrics

metrics = CalibrationMetrics(class_to_calculate=1)

metrics.calculate_metrics(
    labels,
    probs,
    metrics='all'
)
```
# Other features
## Confidence intervals

In addition to point estimates of calibration performance, `calzone` also provides functionality to compute confidence intervals for all metrics. For most metrics, this is computed through bootstrapping. The only exception is the confidence intervals from the reliability diagram which calculates Wilson’s score intervals. The user can specify the number of bootstrap samples and the confidence level. 
```python
from calzone.metrics import CalibrationMetrics

metrics = CalibrationMetrics(class_to_calculate=1)

CalibrationMetrics.bootstrap(
    labels,
    probs,
    metrics='all',
    n_samples=1000
)
```
and a structured NumPy array will be returned.

## Subgroup analysis
`calzone` will perform subgroup analysis by default in the command line user interface. If the user input CSV file contains a subgroup column, the program will compute metrics for the entire dataset and for each subgroup. A detailed description of the input format can be found in the documentation.

## Prevalence adjustment
`calzone` also provides prevalence adjustment to account for prevalence changes between training data and testing data. Since calibration is defined using posterior probability, a mere shift in the disease prevalence of the testing data will result in miscalibration. It can be fixed by searching for the optimal derived original prevalence such that the adjusted probability minimizes a proper scoring rule such as cross-entropy loss. The formula of prevalence adjusted probability is:
$$
P'(D=1|\hat{p}=p) = \frac{\eta'/(1-\eta')}{(1/p-1)(\eta/(1-\eta))} = p'
$$
where $\eta$ is the prevalence of the testing data, $\eta'$ is the prevalence of the training data, and $p$ is the predicted probability [@weijie_prevalence_adjustment;@prevalence_shift;@gu_likelihod_ratio;@Prevalence_HORSCH]. We search for the optimal $\eta'$ that minimizes the cross-entropy loss. The user can also specify $\eta'$ and adjust the probability output directly if the training set prevalence is available.

## Multiclass extension
`calzone` provides a multiclass extension for multiclass classification. The user can specify the class to calculate the metrics using a 1-vs-rest approach and test the calibration of each class. Alternatively, the user can transform the data and make the problem become a top-class calibration problem. The top-class calibration has a similar format to binary classification, but the class 1 probability is defined as the probability of the class with the highest probability and the class 0 probability is defined as 1 minus the probability of the class with the highest probability. The labels are transformed into whether the predicted class equals the true highest probability class, 0 if not and 1 if yes. Notice that the interpretation of some metrics may change in the top-class transformation because the probability of class 1 after transformation is not tied to a specific class in the original dataset.

## Verification of methods
We compared the results calculated by `calzone` with external packages for some metrics to ensure the correctness of the implementation. For the reliability diagram verification, we compared the result with the `sklearn.calibration.calibration_curve()` function in `scikit-learn` [@scikit] with the reliability diagram produced by calzone. For the top-class ECE and SpiegelHalter's Z scores, we compared the result with the `MAPIE` package [@taquet2022mapie]. For the Hosmer-Lemeshow statistic, we compared the result with the `ResourceSelection` package in R language [@ResourceSelection]. The difference in result for all functions tested were within 0.1%, indicating calzone output values are very consistent with the values produced by other packages. We include the verification codes and comparison in our documentation.

## Command line interface
`calzone` also provides a command line interface. Users can visualize the calibration curve, calculate calibration metrics and their confidence intervals using the this. For help on running this functionality, the user can run `python cal_metrics.py -h`.

# Acknowledgements
The mention of commercial products, their sources, or their use in connection with material reported herein is not to be construed as either an actual or implied endorsement of such products by the Department of Health and Human Services. This is a contribution of the U.S. Food and Drug Administration and is not subject to copyright.

The authors acknowledge the Research Participation Program at the Center for Devices and Radiological Health administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and the U.S. Food and Drug Administration. 

# Conflicts of interest
The authors declare no conflicts of interest.

# References
