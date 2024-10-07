"""Visualization functions for the Calibration Measure package."""

__author__ = "Kwok Lung Jason Fan"
__copyright__ = "Copyright 2024"
__credits__ = ["Kwok Lung Jason Fan", "Qian Cao"]
__license__ = "Apache License 2.0"
__version__ = "0.1"
__maintainer__ = "Kwok Lung Jason Fan"
__email__ = "kwoklung.fan@fda.hhs.gov"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt


def plot_reliability_diagram(
    reliabilities,
    confidences,
    bin_counts,
    bin_edges=None,
    line=True,
    error_bar=False,
    z=1.96,
    title="Reliability Diagram",
    save_path=None,
    return_fig=False,
    custom_colors=None,
    dpi=150,
):
    """Plot a reliability diagram to visualize the calibration of a model.

    Args:
        reliabilities (array-like): Empirical frequencies for each bin.
        confidences (array-like): Mean predicted probabilities for each bin.
        bin_counts (array-like): Number of samples in each bin.
        bin_edges (array-like, optional): Edges of the bins. If None, assumes equal-spaced bins.
        line (bool, optional): If True, plot lines connecting points. If False, plot as a bar chart. Defaults to True.
        error_bar (bool, optional): If True, add error bars to the plot. Defaults to False.
        z (float, optional): Z-score for calculating Wilson score interval. Defaults to 1.96.
        title (str, optional): Title of the plot. Defaults to 'Reliability Diagram'.
        save_path (str, optional): Path to save the figure. If None, figure is not saved. Defaults to None.
        return_fig (bool, optional): If True, return the figure object. Defaults to False.
        custom_colors (list, optional): List of custom colors for multi-class plots. Defaults to None.
        dpi (int, optional): DPI for saving the figure. Defaults to 150.

    Returns:
        matplotlib.figure.Figure, optional: The figure object if return_fig is True.
    """

    # Create figure
    fig = plt.figure(figsize=(8, 6))

    # Convert inputs to numpy arrays
    reliabilities = np.array(reliabilities)
    confidences = np.array(confidences)
    bin_counts = np.array(bin_counts)

    # Reshape inputs if they are 1D
    if reliabilities.ndim == 1:
        reliabilities = reliabilities.reshape(1, -1)
        confidences = confidences.reshape(1, -1)
        bin_counts = bin_counts.reshape(1, -1)

    # Set up colors for plotting
    num_classes = reliabilities.shape[0]
    if custom_colors is not None:
        colors = custom_colors
    elif num_classes == 1:
        colors = ["black"]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    # Plot for each class
    for class_idx in range(num_classes):
        class_reliabilities = reliabilities[class_idx].flatten()
        class_confidences = confidences[class_idx].flatten()
        class_bin_counts = bin_counts[class_idx].flatten()
        mask = class_bin_counts > 0

        if line is True:
            # Plot as line
            plt.plot(
                class_confidences[mask],
                class_reliabilities[mask],
                "x",
                color=colors[class_idx],
            )
            plt.plot(
                class_confidences[mask],
                class_reliabilities[mask],
                "-",
                color=colors[class_idx],
                label="all" if num_classes == 1 else f"Class {class_idx}",
            )
        else:
            # Plot as bar chart
            if bin_edges is None:
                bin_edges = np.linspace(
                    0, 1, len(class_bin_counts) + 1
                )  # assume equal spaced bin
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = 1 / (len(bin_centers) * num_classes)
            plt.bar(
                bin_centers + width * class_idx,
                class_reliabilities,
                width=width,
                edgecolor="black",
                align="center",
                alpha=0.7,
                color=colors[class_idx],
                label="all" if num_classes == 1 else f"Class {class_idx}",
            )

        if error_bar is not False:
            # Calculate and plot error bars using Wilson score interval
            n = class_bin_counts[class_bin_counts > 0]
            p_hat = class_reliabilities[mask]
            base = (p_hat + (z**2) / (2 * n)) * (1 / (1 + (z**2) / n))
            plus_minus = (
                (z / (2 * n))
                * (np.sqrt(4 * n * p_hat * (1 - p_hat) + z**2))
                * (1 / (1 + (z**2) / n))
            )
            score_interval = np.zeros((2, len(class_reliabilities[mask])))
            score_interval[0, :] = base - plus_minus
            score_interval[1, :] = base + plus_minus
            score_interval = np.abs(score_interval - class_reliabilities[mask])
            plt.errorbar(
                class_confidences[mask],
                class_reliabilities[mask],
                yerr=score_interval,
                fmt="o",
                color=colors[class_idx],
                capsize=5,
                alpha=0.7,
            )

    # Plot the diagonal line for perfect calibration
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    # Set plot limits and labels
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)

    # Return the figure object or display the plot
    if return_fig:
        return fig
    else:
        plt.show()


def plot_roc_curve(
    fpr,
    tpr,
    roc_auc,
    class_to_plot=None,
    title="ROC Curve",
    save_path=None,
    dpi=150,
    return_fig=False,
):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (array-like): False Positive Rate values.
        tpr (array-like): True Positive Rate values.
        roc_auc (float or array-like): Area Under the ROC Curve (AUC) value(s).
        class_to_plot (int, optional): The class to plot. If None, plots all classes. Defaults to None.
        title (str, optional): Title of the plot. Defaults to 'ROC Curve'.
        save_path (str, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
        dpi (int, optional): The resolution in dots per inch for saving the figure. Defaults to 150.
        return_fig (bool, optional): If True, returns the figure object instead of displaying it. Defaults to False.

    Returns:
        matplotlib.figure.Figure or None: The figure object if return_fig is True, otherwise None.

    This function creates a matplotlib figure showing the ROC curve(s).
    """
    fig = plt.figure(figsize=(8, 6))
    if class_to_plot is not None:
        # Plot ROC curve for a single class
        plt.plot(fpr, tpr, label=f"Class {class_to_plot} (AUC = {roc_auc:.2f})")
    else:
        # Plot ROC curves for all classes
        for i in range(len(tpr)):
            j = i + 1
            plt.plot(fpr[i], tpr[i], label=f"Class {j} (AUC = {roc_auc[i]:.2f})")

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)

    # Return the figure object or display the plot
    if return_fig:
        return fig
    else:
        plt.show()
