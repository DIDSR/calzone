"""Uitlity functions for the Calibration Measure package."""

__author__ = "Kwok Lung Jason Fan"
__copyright__ = "Copyright 2024"
__credits__ = ["Kwok Lung Jason Fan", "Qian Cao"]
__license__ = "Apache License 2.0"
__version__ = "0.1"
__maintainer__ = "Kwok Lung Jason Fan"
__email__ = "kwoklung.fan@fda.hhs.gov"
__status__ = "Development"

import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import copy


def make_roc_curve(y_true, y_proba, class_to_plot=None):
    """
    Compute the Receiver Operating Characteristic (ROC) curve for binary or multiclass classification.

    Args:
        y_true (array-like): True labels of the data. Shape (n_samples,).
        y_proba (array-like): Predicted probabilities of the positive class. Shape (n_samples, n_classes).
        class_to_plot (int, optional): The class to plot the ROC curve for. If None, plots the ROC curve for each class. Default is None.

    Returns:
        fpr (array): False Positive Rate for the selected class or each class. Shape (n_points,).
        tpr (array): True Positive Rate for the selected class or each class. Shape (n_points,).
        roc_auc (float or array): Area Under the ROC Curve (AUC) for the selected class or each class. If class_to_plot is not None, returns a float. If class_to_plot is None, returns an array of shape (n_classes,).

    Note:
        - The input arrays y_true and y_proba must have the same number of samples.
        - The input array y_proba must have probabilities for each class in a multiclass problem.
        - The input array y_proba must not contain any NaN values.

    Example:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_proba = [[0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]
        >>> fpr, tpr, roc_auc = roc_curve(y_true, y_proba, class_to_plot=1)
    """

    # Remove NaN values from input data
    mask = ~np.isnan(y_true) & ~np.isnan(y_proba).any(axis=1)
    y_true_clean = y_true[mask]
    y_proba_clean = y_proba[mask]

    def calculate_roc(y_true, y_score):
        thresholds = np.unique(y_score)
        thresholds = np.concatenate(([thresholds[0] - 1], thresholds))

        tpr = np.zeros(thresholds.shape)
        fpr = np.zeros(thresholds.shape)

        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)

        for i, threshold in enumerate(thresholds):
            y_pred = y_score >= threshold
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tpr[i] = tp / pos
            fpr[i] = fp / neg

        return fpr, tpr

    def calculate_auc(fpr, tpr):
        return np.trapz(tpr, fpr)

    if class_to_plot is not None:
        # Plot ROC for a single class
        fpr, tpr = calculate_roc(
            y_true_clean == class_to_plot, y_proba_clean[:, class_to_plot]
        )
        roc_auc = calculate_auc(fpr, tpr)
    else:
        # Plot ROC for each class
        fpr = []
        tpr = []
        roc_auc = []
        for i in range(y_proba_clean.shape[1]):
            fpr_temp, tpr_temp = calculate_roc(y_true_clean == i, y_proba_clean[:, i])
            roc_auc_temp = calculate_auc(fpr_temp, tpr_temp)
            fpr.append(fpr_temp)
            tpr.append(tpr_temp)
            roc_auc.append(roc_auc_temp)
    return fpr, tpr, roc_auc


def reliability_diagram(
    y_true,
    y_proba,
    num_bins=10,
    class_to_plot=None,
    is_equal_freq=False,
    save_path=None,
):
    """
    Compute the reliability diagram for a binary or multi-class classification model.

    Args:
        y_true (array-like): True labels of the samples. Can be a binary array or a one-hot encoded array.
        y_proba (array-like): Predicted probabilities for each class. Shape should be (n_samples, n_classes).
        num_bins (int): Number of bins to divide the predicted probabilities into. Default is 10.
        class_to_plot (int or None): Index of the class to plot the reliability diagram for. If None, the diagram will be computed for all classes. Default is None.
        is_equal_freq (bool): If True, the bins will be equally frequent. If False, the bins will be equally spaced. Default is False.

    Returns:
        reliabilities (array-like): Array of accuracies for each bin. Shape depends on the value of class_to_plot.
        confidences (array-like): Array of average confidences for each bin. Shape depends on the value of class_to_plot.
        bin_edges (array-like): Array of bin edges.
        bin_counts (array-like): Array of counts for each bin.

    Note:
        - The reliability diagram is a graphical tool to assess the calibration of a classification model. It plots the average predicted probability against the observed accuracy for each bin of predicted probabilities.
        - If y_true is a binary array, it will be converted to a one-hot encoded array internally.
        - If class_to_plot is not None, the reliability diagram will be computed only for the specified class. Otherwise, it will be computed for all classes.
        - The number of bins determines the granularity of the reliability diagram. Higher values result in more bins and a more detailed diagram.
    """

    # Determine bin edges based on equal frequency or equal spacing
    if is_equal_freq:
        quatiles = np.linspace(0, 1, num_bins + 1)
        if class_to_plot is not None:
            bin_edges = np.quantile(y_proba[:, class_to_plot], quatiles)
        else:
            highest_prob = np.max(y_proba, axis=1)
            bin_edges = np.quantile(highest_prob, quatiles)
    else:
        bin_edges = np.linspace(0, 1, num_bins + 1)

    y_true = y_true.flatten()

    if class_to_plot is not None:
        # Compute reliability diagram for a single class
        bin_mask = (
            (y_proba[:, class_to_plot] > bin_edges[:-1, np.newaxis])
            & (y_proba[:, class_to_plot] <= bin_edges[1:, np.newaxis])
        ).T
        bin_counts = np.sum(bin_mask, axis=0)
        bin_accuracies = np.zeros_like(bin_counts, dtype=float)
        bin_confidences = np.zeros_like(bin_counts, dtype=float)
        bin_correct = np.sum(
            (y_true == class_to_plot)[:, np.newaxis] * bin_mask, axis=0
        )
        bin_accuracies = np.divide(
            bin_correct,
            bin_counts,
            out=np.zeros_like(bin_correct, dtype=float),
            where=bin_counts != 0,
        )
        bin_probability = np.sum(
            y_proba[:, class_to_plot][:, np.newaxis] * bin_mask, axis=0
        )
        bin_confidences = np.divide(
            bin_probability,
            bin_counts,
            out=np.zeros_like(bin_probability, dtype=float),
            where=bin_counts != 0,
        )
    else:
        # Compute reliability diagram for all classes
        highest_prob = np.max(y_proba, axis=1)
        bin_mask = (highest_prob[:, np.newaxis] > bin_edges[:-1]) & (
            highest_prob[:, np.newaxis] <= bin_edges[1:]
        )
        bin_counts = np.sum(bin_mask, axis=0)
        y_pred = np.argmax(y_proba, axis=1)
        bin_correct = np.sum((y_true == y_pred)[:, np.newaxis] * bin_mask, axis=0)
        bin_accuracies = np.zeros_like(bin_counts, dtype=float)
        bin_confidences = np.zeros_like(bin_counts, dtype=float)
        bin_probability = np.sum(highest_prob[:, np.newaxis] * bin_mask, axis=0)
        bin_accuracies[bin_counts > 0] = (
            bin_correct[bin_counts > 0] / bin_counts[bin_counts > 0]
        )
        bin_confidences[bin_counts > 0] = (
            bin_probability[bin_counts > 0] / bin_counts[bin_counts > 0]
        )

    if save_path is not None:
        # Save reliability diagram output
        output = np.vstack(
            (bin_accuracies, bin_confidences, bin_edges[1:], bin_counts)
        ).T
        np.savetxt(
            save_path,
            output,
            delimiter=",",
            header="Accuracy,Confidence,Bin_right_edge,Bin_Count",
            comments="",
        )
    return bin_accuracies, bin_confidences, bin_edges, bin_counts


def softmax_to_logits(probabilities, epsilon=1e-7):
    """
    Convert softmax probabilities to logits.

    Args:
        probabilities (array-like): Input probabilities.
        epsilon (float): Small value to avoid log(0). Default is 1e-7.

    Returns:
        numpy.ndarray: Computed logits.
    """
    # Ensure the input is a numpy array
    probabilities = np.array(probabilities)

    # Clip probabilities to avoid log(0)
    clipped_probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # Compute the log of the clipped probabilities
    log_probabilities = np.log(clipped_probabilities)

    # Compute the constant term (which is the log of the sum of exponentials)
    constant = np.log(np.sum(np.exp(log_probabilities)))

    # Compute logits
    logits = log_probabilities + constant

    return logits


def removing_nan(y_true, y_predict, y_proba):
    """
    Remove rows containing NaN values from input arrays.

    Args:
        y_true (array-like): True labels.
        y_predict (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities.

    Returns:
        y_true (array-like): Cleaned version of y_true with NaN rows removed.
        y_predict (array-like): Cleaned version of y_predict with NaN rows removed.
        y_proba (array-like): Cleaned version of y_proba with NaN rows removed.
    """
    normal_row = np.logical_not(np.clip(np.isnan(y_proba).sum(axis=1), 0, 1))
    return y_true[normal_row], y_predict[normal_row], y_proba[normal_row]


def apply_prevalence_adjustment(
    adjusted_prevalence, y_true, y_proba, class_to_calculate=1
):
    """
    Apply the prevalence adjustment method.

    Args:
        adjusted_prevalence (float): The adjusted prevalence to test.
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities.
        class_to_calculate (int): The class index to adjust. Default is 1.

    Returns:
        numpy.ndarray: Adjusted probabilities.
    """
    original_prevalence = np.mean(y_true)
    proba = y_proba[:, class_to_calculate]
    LR = (proba / (1 - proba)) * ((1 - adjusted_prevalence) / adjusted_prevalence)
    adjusted_y_proba = (original_prevalence * LR) / (
        original_prevalence * LR + (1 - original_prevalence)
    )
    other_class_proba = 1 - adjusted_y_proba
    new_y_proba = np.hstack(
        (other_class_proba.reshape(-1, 1), adjusted_y_proba.reshape(-1, 1))
    )
    return new_y_proba


def loss(adjusted_prevalence, y_true, y_proba, class_to_calculate=1):
    """
    Calculate the loss function for prevalence adjustment.

    Args:
        adjusted_prevalence (float): The adjusted prevalence.
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities.
        class_to_calculate (int): The class index to calculate loss for. Default is 1.

    Returns:
        float: Calculated loss value.
    """
    adjusted_y_proba = apply_prevalence_adjustment(
        adjusted_prevalence, y_true, y_proba, class_to_calculate
    )
    loss = -np.mean(
        y_true * np.log(adjusted_y_proba[:, class_to_calculate])
        + (1 - y_true) * np.log(1 - adjusted_y_proba[:, class_to_calculate])
    )
    return loss


def find_optimal_prevalence(y_true, y_proba, class_to_calculate=1, epsilon=1e-7):
    """
    Find the optimal adjustment prevalence using scipy.optimize.

    Args:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities.
        class_to_calculate (int): The class index to optimize for. Default is 1.
        epsilon (float): Small value to avoid numerical instability. Default is 1e-7.

    Returns:
        optimal_prevalence (float): The optimal prevalence.
        adjusted_probabilities (numpy.ndarray): The adjusted probabilities using the optimal prevalence.
    """
    y_proba = np.clip(y_proba, epsilon, 1 - epsilon)

    def objective(adjusted_prevalence):
        return loss(adjusted_prevalence, y_true, y_proba, class_to_calculate)

    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    return result.x, apply_prevalence_adjustment(
        result.x, y_true, y_proba, class_to_calculate
    )



def transform_topclass(probs, labels):
    """
    Transforms the data to top class binary problem

    Args:
        probs (numpy.ndarray): Array of probability values
        labels (numpy.ndarray): Array of label values

    Returns:
        tuple: (transformed_probs, transformed_labels)
    """
    top_class = np.argmax(probs, axis=1)
    transformed_probs = np.column_stack(
        (1 - np.max(probs, axis=1), np.max(probs, axis=1))
    )
    transformed_labels = (
        (labels.flatten() == top_class).astype(int).reshape(-1, 1)
    )
    return transformed_probs, transformed_labels
class data_loader:
    """
    A class for loading and preprocessing data from a CSV file.

    This class handles various data formats, including those with or without subgroup information and headers.

    Attributes:
        data_path (str): Path to the CSV file containing the data.
        Header (numpy.ndarray): Array of column headers from the CSV file.
        subgroups (list): List of subgroup column names, if present.
        subgroup_indices (list): List of indices for subgroup columns, if present.
        have_subgroup (bool): Flag indicating whether subgroup information is present.
        data (numpy.ndarray): Raw data loaded from the CSV file.
        probs (numpy.ndarray): Probability values extracted from the data.
        labels (numpy.ndarray): Label values extracted from the data.
        subgroups_class (list): List of unique subgroup classes for each subgroup, if present.
        subgroups_index (list): List of indices for each subgroup class, if present.

    Methods:
        __init__(self, data_path): Initializes the data_loader object and loads data from a CSV file.
        transform_topclass(self): Transforms the data to top class binary problem.
    """

    def __init__(self, data_path):
        """
        Initializes the data_loader object and loads data from the specified file.

        Args:
            data_path (str): Path to the CSV file containing the data.

        The method performs the following steps:
        1. Loads the header from the CSV file.
        2. Checks for the presence of subgroup information.
        3. Loads the data based on the presence or absence of subgroup information.
        4. Extracts probability values and labels from the loaded data.
        5. If subgroups are present, extracts subgroup classes and their indices.

        Note:
        - If there is a header, it must be in the format: proba_0,proba_1,...,subgroup_1(optional),subgroup_2(optional),...,label
        - If there is no header, the columns must be in the order of proba_0,proba_1,...,label
        """
        self.data_path = data_path
        self.Header = np.loadtxt(self.data_path, delimiter=",", max_rows=1, dtype=str)
        if any("subgroup" in col for col in self.Header):
            self.subgroups = [col for col in self.Header if "subgroup" in col]
            self.subgroup_indices = [
                i for i, col in enumerate(self.Header) if "subgroup" in col
            ]
            self.have_subgroup = True
        else:
            self.have_subgroup = False
        if not self.have_subgroup:
            if self.Header[-1] == "label":
                self.data = np.loadtxt(self.data_path, delimiter=",", skiprows=1)
                self.probs = self.data[:, :-1]
                self.labels = self.data[:, -1:].astype(int)
            else:
                self.data = np.loadtxt(self.data_path, delimiter=",")
                self.probs = self.data[:, :-1].astype(float)
                self.labels = self.data[:, -1:].astype(int)
        else:
            self.data = np.loadtxt(self.data_path, delimiter=",", skiprows=1, dtype=str)
            self.probs = self.data[:, : -len(self.subgroups) - 1].astype(float)
            self.labels = self.data[:, -1:].astype(int)
            self.subgroups_class = []
            self.subgroups_index = []
            for i, subgroup in enumerate(self.subgroups):
                self.subgroups_class.append(
                    np.unique(self.data[:, self.subgroup_indices[i]])
                )
                indices = []
                for j, subgroup_class in enumerate(self.subgroups_class[i]):
                    indices.append(
                        np.where(
                            self.data[:, self.subgroup_indices[i]] == subgroup_class
                        )[0]
                    )
                self.subgroups_index.append(indices)

    def transform_topclass(self):
        """
        Transforms the data to top class binary problem

        Returns:
            data_loader: A new data_loader object with transformed data
        """
        new_loader = copy.deepcopy(self)
        top_class = np.argmax(self.probs, axis=1)
        new_loader.probs = np.column_stack(
            (1 - np.max(self.probs, axis=1), np.max(self.probs, axis=1))
        )
        new_loader.labels = (
            (self.labels.flatten() == top_class).astype(int).reshape(-1, 1)
        )
        new_loader.data = np.column_stack((new_loader.probs, new_loader.labels))
        return new_loader

class fake_binary_data_generator:
    """A class for generating fake binary data and applying miscalibration.

    This class provides methods to generate binary classification data
    and apply different types of miscalibration to the probabilities.

    Attributes:
        alpha_val (float): Alpha parameter for the beta distribution.
        beta_val (float): Beta parameter for the beta distribution.
    """

    def __init__(self, alpha_val, beta_val):
        """Initialize the fake binary data generator.

        Args:
            alpha_val (float): Alpha parameter for the beta distribution.
            beta_val (float): Beta parameter for the beta distribution.
        """
        self.alpha_val = alpha_val
        self.beta_val = beta_val

    def generate_data(self, sample_size):
        """Generate fake binary classification data.

        Args:
            sample_size (int): Number of samples to generate.

        Returns:
            X (numpy.ndarray): Array of shape (sample_size, 2) containing probabilities for each class.
            y_true (numpy.ndarray): Array of shape (sample_size,) containing true binary labels.
        """
        # Generate probabilities for class 1 using beta distribution
        class1_proba = beta.rvs(self.alpha_val, self.beta_val, size=sample_size)

        # Calculate probabilities for class 0
        class0_proba = 1 - class1_proba

        # Combine probabilities for both classes
        X = np.concatenate(
            (class0_proba.reshape(-1, 1), class1_proba.reshape(-1, 1)), axis=1
        )

        # Generate true labels using binomial distribution
        y_true = np.random.binomial(1, p=class1_proba)

        return X, y_true

    def linear_miscal(self, X, miscal_scale):
        """Apply linear miscalibration to the input probabilities.

        This function transforms the input probabilities to logits,
        applies a linear scaling, and then converts back to probabilities.

        Args:
            X (numpy.ndarray): Input probabilities of shape (n_samples, 2).
            miscal_scale (float): Scale factor for miscalibration.

        Returns:
            numpy.ndarray: Miscalibrated probabilities of shape (n_samples, 2).
        """
        # Convert input probabilities to logits
        logits = softmax_to_logits(X)

        # Apply linear scaling to logits
        miscal_logits = logits * miscal_scale

        # Convert scaled logits back to probabilities
        miscal_probs = softmax(miscal_logits, axis=1)

        return miscal_probs

    def abraitary_miscal(self, logits, miscal_function):
        """Apply arbitrary miscalibration to the input logits.

        This function allows for the application of any custom miscalibration
        function to the input logits.

        Args:
            logits (numpy.ndarray): Input logits of shape (n_samples, 2).
            miscal_function (callable): Function to apply miscalibration to the logits.

        Returns:
            numpy.ndarray: Miscalibrated probabilities of shape (n_samples, 2).
        """
        # Apply the custom miscalibration function to the logits
        miscal_logits = miscal_function(logits)

        # Convert miscalibrated logits to probabilities
        miscal_probs = softmax(miscal_logits, axis=1)

        return miscal_probs
