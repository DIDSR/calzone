"""Metrics calculation functions for the Calibration Measure package."""

__author__ = "Kwok Lung Jason Fan"
__copyright__ = "Copyright 2024"
__credits__ = ["Kwok Lung Jason Fan", "Qian Cao"]
__license__ = "Apache License 2.0"
__version__ = "0.1"
__maintainer__ = "Kwok Lung Jason Fan"
__email__ = "kwoklung.fan@fda.hhs.gov"
__status__ = "Development"
import scipy.stats as stats
import numpy as np
from .utils import *
import statsmodels.api as sm
import statsmodels.nonparametric.smoothers_lowess as lowess
import numpy.lib.recfunctions as rf


def hosmer_lemeshow_test(reliability, confidence, bin_count):
    """
    Compute the Hosmer-Lemeshow test for goodness of fit.

    This test is used to assess the calibration of binary classification models with full probability outputs.
    It compares observed and expected frequencies of events in groups of the data.

    Args:
        reliability (array-like): Observed proportion of positive samples in each bin.
        confidence (array-like): Predicted probabilities for each bin.
        bin_count (array-like): Number of samples in each bin.

    Returns:
        chi_squared (float): The chi-squared statistic of the Hosmer-Lemeshow test.
        p_value (float): The p-value associated with the chi-squared statistic.
        df (int): The degrees of freedom for the test.

    Note:
        - The Hosmer-Lemeshow test is widely used for assessing calibration in probabiliticst models.
        - A small p-value (typically < 0.05) suggests that the model is a poor fit to the data.
        - This test can be sensitive to the number of groups and sample size.
        - It is recommended to use the Hosmer-Lemeshow test in conjunction with other metrics.
    """

    # Convert inputs to numpy arrays for consistent handling
    reliability = np.array(reliability)
    confidence = np.array(confidence)
    bin_count = np.array(bin_count)

    # Remove bins with zero count to avoid division by zero
    mask = bin_count != 0
    reliability = reliability[mask]
    confidence = confidence[mask]
    bin_count = bin_count[mask]

    # Compute observed and expected values for each bin
    observed = reliability * bin_count
    expected = confidence * bin_count

    # Compute the Hosmer-Lemeshow statistic
    chi_squared = np.sum(
        (observed - expected) ** 2 / (expected * (1 - expected / bin_count))
    )

    # Compute degrees of freedom and p-value
    num_bins = len(bin_count)
    df = num_bins - 2  # Subtract 2 for the parameters estimated in the model
    p_value = 1 - stats.chi2.cdf(chi_squared, df)

    return chi_squared, p_value, df


def calculate_ece_mce(reliability, confindence, bin_counts):
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    These metrics assess the calibration of a classification model by comparing
    predicted probabilities to observed frequencies.

    Args:
        reliability (array-like): Array of observed frequencies for each bin.
        confindence (array-like): Array of predicted probabilities for each bin.
        bin_counts (array-like): Array of sample counts in each bin.

    Returns:
        ece (float): The Expected Calibration Error.
        mce (float): The Maximum Calibration Error.

    Note:
        - ECE is a weighted average of the absolute differences between confidence and reliability.
        - MCE is the maximum absolute difference between confidence and reliability across all bins.
        - Lower values of both ECE and MCE indicate better calibration.
    """

    # Calculate total number of samples
    total_count = np.sum(bin_counts)

    # Compute absolute difference between reliability and confidence
    error = np.abs(reliability - confindence)

    # Compute Maximum Calibration Error
    mce = np.max(error)

    # Compute Expected Calibration Error
    ece = np.sum(bin_counts * error) / total_count

    return ece, mce


def spiegelhalter_z_test(y_true, y_proba, class_to_calculate=1):
    """
    Perform Spiegelhalter's Z-test for calibration of probabilistic predictions.

    This test assesses whether predicted probabilities are well-calibrated by comparing
    them to observed outcomes.

    Args:
        y_true (array-like): True labels of the samples.
        y_proba (array-like): Predicted probabilities for each class. Shape should be (n_samples, n_classes).
        class_to_calculate (int): Index of the class to calculate the test for. Default is 1.

    Returns:
        z_score (float): The z-score of the Spiegelhalter's Z-test.
        p_value (float): The p-value associated with the z-score.

    Note:
        - This test is used to assess the calibration of a classification model.
        - A small p-value (typically < 0.05) suggests that the model is poorly calibrated.
        - The test assumes that predictions are independent and identically distributed.
    """
    # Ensure inputs are numpy arrays and flatten y_true
    y_true = np.array(y_true).ravel()
    y_proba = np.array(y_proba)

    # Convert to binary classification problem
    mask = y_true == class_to_calculate
    y_true[mask] = 1
    y_true[np.logical_not(mask)] = 0

    # Calculate components of the test statistic
    numerator = np.sum(
        (y_true - y_proba[:, class_to_calculate])
        * (1 - 2 * y_proba[:, class_to_calculate])
    )
    denominator = np.sqrt(
        np.sum(
            ((1 - 2 * y_proba[:, class_to_calculate]) ** 2)
            * y_proba[:, class_to_calculate]
            * (1 - y_proba[:, class_to_calculate])
        )
    )

    # Compute z-score and p-value
    z_score = numerator / denominator
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

    return z_score, p_value


def cox_regression_analysis(
    y_true,
    y_proba,
    epsilon=1e-10,
    class_to_calculate=1,
    print_results=False,
    fix_intercept=False,
    fix_slope=False,
    **kwargs,
):
    """
    Perform Cox regression analysis for classification calibration.

    This function fits a logistic regression model to the logit of predicted probabilities
    to assess the calibration of classification predictions.

    Args:
        y_true (array-like): True binary labels. If multi-class, will be converted to binary.
        y_proba (array-like): Predicted probabilities. Should be of shape (n_samples, n_classes).
        epsilon (float): Small value to avoid log(0) errors. Default is 1e-10.
        class_to_calculate (int): The class to treat as the positive class in binary classification. Default is 1.
        print_results (bool): If True, prints the summary of the logistic regression results. Default is False.
        fix_intercept (bool): If True, fixes the intercept to 0. Can't be used with fix_slope. Default is False.
        fix_slope (bool): If True, fixes the coefficient to 1. Can't be used with fix_intercept. Default is False.

    Returns:
        coef (float): The coefficient (slope) of the Cox regression.
        intercept (float): The intercept of the Cox regression.
        coef_ci (tuple): The confidence interval for the coefficient.
        intercept_ci (tuple): The confidence interval for the intercept.

    Note:
        - A well-calibrated model should have a coefficient close to 1 and an intercept close to 0.
        - The function clips probabilities to avoid numerical instability.
        - For multi-class problems, the function converts the problem to binary classification
          based on the specified class_to_calculate.
    """
    # Clip probabilities to avoid numerical issues
    proba = np.clip(y_proba[:, 1], epsilon, 1 - epsilon)

    # Convert to binary classification problem
    y_true = np.array(y_true).ravel()
    mask = y_true == class_to_calculate
    y_true[mask] = 1
    y_true[np.logical_not(mask)] = 0

    # Calculate logit of probabilities
    logit = np.log(proba / (1 - proba))

    # Add constant term to predictor variables
    X = sm.add_constant(logit)

    # Fit the logistic regression model
    logit_model = sm.Logit(y_true, X)
    if fix_intercept == True:
        logit_result = logit_model.fit_constrained("const=0", disp=0)
    elif fix_slope == True:
        logit_result = logit_model.fit_constrained("x1=1", disp=0)
    else:
        logit_result = logit_model.fit(disp=0)

    # Print results if requested
    if print_results:
        print(logit_result.summary())

    # Extract and return coefficients and confidence intervals
    intercept = logit_result.params[0]
    intercept_ci = logit_result.conf_int()[0, :]
    coef = logit_result.params[1]
    coef_ci = logit_result.conf_int()[1, :]

    return coef, intercept, coef_ci, intercept_ci


def cal_ICI_cox(
    coef, intercept, y_proba, class_to_calculate=1, epsilon=1e-10, **kwargs
):
    """
    Calculate the Integrated Calibration Index (ICI) for a given Cox regression model.

    The ICI measures the average absolute difference between the predicted probabilities
    and the probabilities transformed by the fitted Cox regression model.

    Args:
        coef (float): The coefficient (slope) from the Cox regression.
        intercept (float): The intercept from the Cox regression.
        y_proba (array-like): Predicted probabilities. Should be of shape (n_samples, n_classes).
        class_to_calculate (int): The class to calculate the ICI for in multi-class problems. Default is 1.
        epsilon (float): Small value to avoid numerical instability when clipping probabilities. Default is 1e-10.

    Returns:
        ICI (float): The Integrated Calibration Index.

    Note:
        - Lower ICI values indicate better calibration.
        - The function applies the inverse logit transformation to the predicted probabilities
          using the coefficients from the Cox regression.
    """
    # Define the transformation function based on Cox regression results
    func = lambda x: 1 / (1 + np.exp(-(coef * np.log(x / (1 - x)) + intercept)))

    # Clip probabilities to avoid numerical issues
    proba_clipped = np.clip(y_proba[:, class_to_calculate], epsilon, 1 - epsilon)

    # Apply the transformation function
    transformed_proba = func(proba_clipped)

    # Calculate the ICI
    ICI = np.mean(np.abs(transformed_proba - proba_clipped))

    return ICI


def lowess_regression_analysis(
    y_true,
    y_proba,
    epsilon=1e-10,
    class_to_calculate=1,
    span=0.5,
    delta=0.001,
    it=0,
    **kwargs,
):
    """
    Perform Lowess regression analysis for classification calibration.

    This function applies Locally Weighted Scatterplot Smoothing (LOWESS) to assess
    the calibration of classification predictions.

    Args:
        y_true (array-like): True binary labels. If multi-class, will be converted to binary.
        y_proba (array-like): Predicted probabilities. Should be of shape (n_samples, n_classes).
        epsilon (float, optional): Small value to avoid numerical instability when clipping probabilities. Defaults to 1e-10.
        class_to_calculate (int, optional): The class to treat as the positive class in binary classification. Defaults to 1.
        span (float, optional): The fraction of the data used when estimating each y-value. Defaults to 0.5.
        delta (float, optional): Distance within which to use linear-interpolation instead of weighted regression. Defaults to 0.001.
        it (int, optional): The number of residual-based reweightings to perform. Defaults to 0.

    Returns:
        ICI (float): The Integrated Calibration Index.
        sorted_proba (array-like): Sorted predicted probabilities.
        smoothed_proba (array-like): Corresponding LOWESS-smoothed actual probabilities.

    Note:
        - The function clips probabilities to avoid numerical instability.
        - For multi-class problems, the function converts the problem to binary classification
          based on the specified class_to_calculate.
        - The Integrated Calibration Index (ICI) provides a measure of calibration error,
          with lower values indicating better calibration.
    """
    # Clip probabilities to avoid numerical issues
    proba = np.clip(y_proba[:, class_to_calculate], epsilon, 1 - epsilon)

    # Convert to binary classification problem
    y_true = np.array(y_true).ravel()
    mask = y_true == class_to_calculate
    y_true[mask] = 1
    y_true[np.logical_not(mask)] = 0

    # Perform LOWESS regression
    lowess_fit = lowess.lowess(
        y_true,
        y_proba[:, class_to_calculate],
        frac=span,
        delta=delta,
        it=it,
        return_sorted=True,
    )

    # Calculate the Integrated Calibration Index (ICI)
    ICI = np.abs(lowess_fit[:, 0] - lowess_fit[:, 1]).mean()

    return ICI, lowess_fit[:, 0], lowess_fit[:, 1]


def cal_ICI(func, y_proba, points=1000, class_to_calculate=1):
    """
    Calculate the Integrated Calibration Index (ICI) for a given calibration function.

    Args:
        func (callable): The calibration function to evaluate.
        y_proba (array-like): Predicted probabilities for each class. Shape (n_samples, n_classes).
        points (int, optional): Number of points to use for numerical integration. Defaults to 1000.
        class_to_calculate (int, optional): The class index to calculate the ICI for. Defaults to 1.

    Returns:
        float: The Integrated Calibration Index (ICI) value.

    Note:
        The ICI is calculated by integrating the absolute difference between
        the calibration function and the identity function, weighted by the
        density of the predicted probabilities.
    """
    # Generate evenly spaced points for integration
    x = np.linspace(0, 1, points)
    xcenter = (x[:-1] + x[1:]) / 2

    # Calculate the density of predicted probabilities
    density = np.histogram(y_proba[:, class_to_calculate], bins=x, density=True)[0]

    # Apply the calibration function
    y = func(xcenter)

    # Calculate and return the ICI
    return np.trapz(np.abs(y - xcenter) * density, xcenter)


def logit_func(coef, intercept):
    """
    Create a logistic function with given coefficient and intercept.

    Args:
        coef (float): The coefficient (slope) of the logistic function.
        intercept (float): The intercept of the logistic function.

    Returns:
        callable: A function that takes an input x and returns the logistic function value.

    Note:
        The returned function applies the logistic transformation:
        f(x) = 1 / (1 + exp(-(coef * log(x / (1 - x)) + intercept)))
    """
    return lambda x: 1 / (1 + np.exp(-(coef * np.log(x / (1 - x)) + intercept)))


def get_CI(result, alpha=0.05):
    """
    Calculate confidence intervals for each field in the result.

    Args:
        result (numpy.ndarray): Structured array containing the results for which to calculate confidence intervals.
        alpha (float, optional): The significance level for the confidence interval calculation. Defaults to 0.05.

    Returns:
        dict: A dictionary where keys are field names from the input array and values
              are tuples containing the lower and upper bounds of the confidence interval.

    Note:
        This function calculates percentile-based confidence intervals for each field
        in the input structured array. It's useful for bootstrap or Monte Carlo simulations.
    """
    CI = {}
    for key in result.dtype.names:
        CI[key] = (
            np.percentile(result[key], 100 * (alpha / 2)),
            np.percentile(result[key], 100 * (1 - alpha / 2)),
        )
    return CI


class CalibrationMetrics:
    """
    A class for calculating calibration metrics for classification models.
    """

    def __init__(self, class_to_calculate=1, num_bins=10):
        """
        Initialize the CalibrationMetrics class.

        Args:
            class_to_calculate (int, optional): The class index to calculate the metrics for. Defaults to 1.
            num_bins (int, optional): Number of bins to use for the ECE/MCE/HL calculations. Defaults to 10.
        """
        self.class_to_calculate = class_to_calculate
        self.num_bins = num_bins

    def calculate_metrics(
        self,
        y_true,
        y_proba,
        metrics,
        perform_pervalance_adjustment=False,
        return_numpy=False,
        **kwargs,
    ):
        """
        Calculate the specified calibration metrics.

        This function computes various calibration metrics for binary classification models.
        It supports multiple metrics and can perform prevalence adjustment if needed.

        List of available metrics:

        * SpiegelhalterZ: Spiegelhalter's Z-test for calibration
        * ECE-H: Expected Calibration Error with equal-space binning
        * MCE-H: Maximum Calibration Error with equal-space binning
        * HL-H: Hosmer-Lemeshow test with equal-space binning
        * ECE-C: Expected Calibration Error with equal-count binning
        * MCE-C: Maximum Calibration Error with equal-count binning
        * HL-C: Hosmer-Lemeshow test with equal-count binning
        * COX: Cox regression analysis
        * Loess: Locally Estimated Scatterplot Smoothing regression analysis

        Args:
            y_true (numpy.ndarray): True labels.
            y_proba (numpy.ndarray): Predicted probabilities.
            metrics (list of str or 'all'): List of metric names to calculate. If 'all', calculates all available metrics.
            perform_pervalance_adjustment (bool, optional): Whether to perform prevalence adjustment. Defaults to False.
            return_numpy (bool, optional): Whether to return the results as a numpy array. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the metric calculation functions.

        Returns:
            dict or numpy.ndarray: A dictionary containing the calculated metrics, or a numpy array if return_numpy is True.
        """
        # Initialize results dictionary and set up metrics list
        if metrics == "all":
            metrics = [
                "SpiegelhalterZ",
                "ECE-H",
                "MCE-H",
                "HL-H",
                "ECE-C",
                "MCE-C",
                "HL-C",
                "COX",
                "Loess",
            ]
        results = {}
        precompute_H = False
        precompute_C = False

        # Perform prevalence adjustment if requested
        if perform_pervalance_adjustment:
            optimal_prevalance, y_proba = find_optimal_prevalence(
                y_true=y_true,
                y_proba=y_proba,
                class_to_calculate=self.class_to_calculate,
            )
            class_to_calculate = 1
            results["Optimal prevalence"] = optimal_prevalance
        else:
            class_to_calculate = self.class_to_calculate

        # Calculate each requested metric
        for metric in metrics:
            # Spiegelhalter's Z-test
            if metric == "SpiegelhalterZ":
                score, p_value = spiegelhalter_z_test(
                    y_true=y_true,
                    y_proba=y_proba,
                    class_to_calculate=class_to_calculate,
                )
                results["SpiegelhalterZ score"] = score
                results["SpiegelhalterZ p-value"] = p_value

            # Metrics using equal-space binning
            elif metric in ["ECE-H", "MCE-H", "HL-H"]:
                if not precompute_H:
                    # Precompute equal-space binning results for efficiency
                    acc_H, confidence_H, bin_edges_H, bin_count_H = reliability_diagram(
                        y_true=y_true,
                        y_proba=y_proba,
                        num_bins=self.num_bins,
                        is_equal_freq=False,
                    )
                    (
                        acc_H_class,
                        confidence_H_class,
                        bin_edges_H_class,
                        bin_count_H_class,
                    ) = reliability_diagram(
                        y_true=y_true,
                        y_proba=y_proba,
                        num_bins=self.num_bins,
                        is_equal_freq=False,
                        class_to_plot=class_to_calculate,
                    )
                    precompute_H = True

                if metric == "ECE-H":
                    ece_h, mce_h = calculate_ece_mce(acc_H, confidence_H, bin_count_H)
                    ece_h_class, mce_h_class = calculate_ece_mce(
                        acc_H_class, confidence_H_class, bin_count_H_class
                    )
                    results["ECE-H topclass"] = ece_h
                    results["ECE-H"] = ece_h_class
                elif metric == "MCE-H":
                    ece_h, mce_h = calculate_ece_mce(acc_H, confidence_H, bin_count_H)
                    ece_h_class, mce_h_class = calculate_ece_mce(
                        acc_H_class, confidence_H_class, bin_count_H_class
                    )
                    results["MCE-H topclass"] = mce_h
                    results["MCE-H"] = mce_h_class
                elif metric == "HL-H":
                    hl_h_score, hl_h, _ = hosmer_lemeshow_test(
                        acc_H_class, confidence_H_class, bin_count_H_class
                    )
                    results["HL-H score"] = hl_h_score
                    results["HL-H p-value"] = hl_h

            # Metrics using equal-count binning
            elif metric in ["ECE-C", "MCE-C", "HL-C"]:
                if not precompute_C:
                    # Precompute equal-count binning results for efficiency
                    acc_C, confidence_C, bin_edges_C, bin_count_C = reliability_diagram(
                        y_true=y_true,
                        y_proba=y_proba,
                        num_bins=self.num_bins,
                        is_equal_freq=True,
                    )
                    (
                        acc_C_class,
                        confidence_C_class,
                        bin_edges_C_class,
                        bin_count_C_class,
                    ) = reliability_diagram(
                        y_true=y_true,
                        y_proba=y_proba,
                        num_bins=self.num_bins,
                        is_equal_freq=True,
                        class_to_plot=class_to_calculate,
                    )
                    precompute_C = True

                if metric == "ECE-C":
                    ece_c, mce_c = calculate_ece_mce(acc_C, confidence_C, bin_count_C)
                    ece_c_class, mce_c_class = calculate_ece_mce(
                        acc_C_class, confidence_C_class, bin_count_C_class
                    )
                    results["ECE-C topclass"] = ece_c
                    results["ECE-C"] = ece_c_class
                elif metric == "MCE-C":
                    ece_c, mce_c = calculate_ece_mce(acc_C, confidence_C, bin_count_C)
                    ece_c_class, mce_c_class = calculate_ece_mce(
                        acc_C_class, confidence_C_class, bin_count_C_class
                    )
                    results["MCE-C topclass"] = mce_c
                    results["MCE-C"] = mce_c_class
                elif metric == "HL-C":
                    hl_c_score, hl_c, _ = hosmer_lemeshow_test(
                        acc_C_class, confidence_C_class, bin_count_C_class
                    )
                    results["HL-C score"] = hl_c_score
                    results["HL-C p-value"] = hl_c

            # Cox regression analysis
            elif metric == "COX":
                coef, intercept, coef_ci, intercept_ci = cox_regression_analysis(
                    y_true=y_true, y_proba=y_proba, **kwargs
                )
                results["COX coef"] = coef
                results["COX intercept"] = intercept
                results["COX coef lowerci"] = coef_ci[0]
                results["COX coef upperci"] = coef_ci[1]
                results["COX intercept lowerci"] = intercept_ci[0]
                results["COX intercept upperci"] = intercept_ci[1]
                ICI = cal_ICI_cox(
                    coef,
                    intercept,
                    y_proba,
                    class_to_calculate=class_to_calculate,
                    **kwargs,
                )
                results["COX ICI"] = ICI

            # Loess regression analysis
            elif metric == "Loess":
                loess_ICI, _, _ = lowess_regression_analysis(
                    y_true, y_proba, class_to_calculate=class_to_calculate, **kwargs
                )
                results["Loess ICI"] = loess_ICI

        # Convert results to numpy array if requested
        if return_numpy:
            results = np.array(list(results.values()))

        return results

    def bootstrap(
        self,
        y_true,
        y_proba,
        metrics,
        perform_pervalance_adjustment=False,
        n_samples=1000,
        **kwargs,
    ):
        """
        Run bootstrap and return a numpy structured array with correct field names.

        This function performs bootstrap resampling to estimate the distribution of calibration metrics.
        It generates multiple samples with replacement from the input data and calculates the specified
        metrics for each sample.

        Args:
            y_true (array-like): True labels.
            y_proba (array-like): Predicted probabilities.
            metrics (list of str): List of metric names to calculate.
            perform_pervalance_adjustment (bool, optional): Whether to perform prevalence adjustment for each bootstrap sample. Defaults to False.
            n_samples (int, optional): Number of bootstrap samples to generate. Defaults to 1000.
            **kwargs: Additional keyword arguments to pass to the metric calculation functions.

        Returns:
            numpy.ndarray: A structured array containing the bootstrapped metrics. Each field in the array
                           corresponds to a metric, and each row represents a bootstrap sample.
        """
        bootstrap_results = []
        n_instances = len(y_true)

        # Generate bootstrap samples and calculate metrics
        for i in range(n_samples):
            # Sample with replacement
            indices = np.random.choice(n_instances, n_instances, replace=True)
            y_true_sample = y_true[indices]
            y_proba_sample = y_proba[indices]

            # Calculate metrics for the first sample to get the keys
            if i == 0:
                sample_results = self.calculate_metrics(
                    y_true_sample,
                    y_proba_sample,
                    metrics,
                    perform_pervalance_adjustment=perform_pervalance_adjustment,
                    return_numpy=False,
                    **kwargs,
                )
                keys = sample_results.keys()
                sample_results = np.array(list(sample_results.values()))
            else:
                sample_results = self.calculate_metrics(
                    y_true_sample,
                    y_proba_sample,
                    metrics,
                    perform_pervalance_adjustment=perform_pervalance_adjustment,
                    return_numpy=True,
                    **kwargs,
                )

            bootstrap_results.append(sample_results)

        # Create a structured array with correct field names
        dtype = [(key, float) for key in keys]
        unstructured_results = np.array(bootstrap_results)
        structured_results = rf.unstructured_to_structured(
            unstructured_results, dtype=dtype
        )

        return structured_results

    def optimal_prevalence_adjustment(self, y_true, y_proba):
        """
        Perform optimal prevalence adjustment and return adjusted probabilities.

        This function finds the optimal prevalence value that minimizes the difference
        between the predicted and actual positive rates, and then adjusts the input
        probabilities accordingly.

        Args:
            y_true (array-like): True labels.
            y_proba (array-like): Predicted probabilities.

        Returns:
            optimal_prevalence (float): Optimal prevalence value.
            adjusted_proba (array-like): Adjusted probabilities. First column is the adjusted probabilities for the other class,
                second column is the adjusted probabilities for the class of interest.
        """
        optimal_prevalence, adjusted_proba = find_optimal_prevalence(
            y_true, y_proba, class_to_calculate=self.class_to_calculate
        )
        return optimal_prevalence, adjusted_proba
