import argparse
import numpy as np
from calzone.metrics import CalibrationMetrics, get_CI
from calzone.utils import *
from calzone.vis import plot_reliability_diagram


def perform_calculation(probs, labels, args, suffix=""):
    """
    Calculate calibration metrics and visualize reliability diagram.

    Args:
        probs (numpy.ndarray): Predicted probabilities for each class.
        labels (numpy.ndarray): True labels.
        args (argparse.Namespace): Command-line arguments.
        suffix (str, optional): Suffix for output files. Defaults to "".

    Returns:
        numpy.ndarray: Calculated metrics and confidence intervals (if bootstrapping is used).
    """
    cal_metrics = CalibrationMetrics(
        class_to_calculate=args.class_to_calculate, num_bins=args.num_bins
    )

    metrics_to_calculate = args.metrics.split(",") if args.metrics else ["all"]
    if metrics_to_calculate == ["all"]:
        metrics_to_calculate = "all"
    result = cal_metrics.calculate_metrics(
        y_true=labels,
        y_proba=probs,
        metrics=metrics_to_calculate,
        perform_pervalance_adjustment=args.prevalence_adjustment,
    )

    keys = list(result.keys())
    result = np.array(list(result.values())).reshape(1, -1)

    if args.n_bootstrap > 0:
        bootstrap_results = cal_metrics.bootstrap(
            y_true=labels,
            y_proba=probs,
            n_samples=args.n_bootstrap,
            metrics=metrics_to_calculate,
            perform_pervalance_adjustment=args.prevalence_adjustment,
        )
        CI = get_CI(bootstrap_results)
        result = np.vstack((result, np.array(list(CI.values())).T))

    if args.verbose:
        print_metrics(result, keys, args.n_bootstrap, suffix)

    if args.save_metrics:
        save_metrics_to_csv(result, keys, args.save_metrics, suffix)

    if args.plot:
        plot_reliability(labels, probs, args, suffix)
    return result


def print_metrics(result, keys, n_bootstrap, suffix):
    """
    Print calculated metrics.

    Args:
        result (numpy.ndarray): Calculated metrics and confidence intervals.
        keys (list): Names of the calculated metrics.
        n_bootstrap (int): Number of bootstrap samples.
        suffix (str): Suffix for output files.
    """
    if n_bootstrap > 0:
        print_header = (
            "Metrics with bootstrap confidence intervals:"
            if suffix == ""
            else f"Metrics for {suffix} with bootstrap confidence intervals:"
        )
        print(print_header)
        for i, num in enumerate(keys):
            print(
                f"{num}: {np.format_float_positional(result[0][i], 3)}",
                f"({np.format_float_positional(result[1][i], 3)}, "
                f"{np.format_float_positional(result[2][i], 3)})",
            )
    else:
        print_header = "Metrics:" if suffix == "" else f"Metrics for subgroup {suffix}:"
        print(print_header)
        for i, num in enumerate(keys):
            print(f"{num}: {np.format_float_positional(result[0][i], 3)}")


def save_metrics_to_csv(result, keys, save_metrics, suffix):
    """
    Save calculated metrics to a CSV file.

    Args:
        result (numpy.ndarray): Calculated metrics and confidence intervals.
        keys (list): Names of the calculated metrics.
        save_metrics (str): Path to save the CSV file.
        suffix (str): Suffix for output files.
    """
    if suffix == "":
        filename = save_metrics
    else:
        split_filename = save_metrics.split(".")
        pathwithoutextension = ".".join(split_filename[:-1])
        filename = pathwithoutextension + "_" + suffix + ".csv"
    np.savetxt(
        filename,
        np.array(result),
        delimiter=",",
        header=",".join(keys),
        comments="",
        fmt="%s",
    )
    print("Result saved to", filename)


def plot_reliability(labels, probs, args, suffix):
    """
    Plot and save reliability diagram.

    Args:
        labels (numpy.ndarray): True labels.
        probs (numpy.ndarray): Predicted probabilities for each class.
        args (argparse.Namespace): Command-line arguments.
        suffix (str): Suffix for output files.
    """
    if suffix == "":
        filename = args.save_plot
        diagram_filename = args.save_diagram_output or None
    else:
        split_filename = args.save_plot.split(".")
        pathwithoutextension = ".".join(split_filename[:-1])
        filename = pathwithoutextension + "_" + suffix + "." + split_filename[-1]
        if args.save_diagram_output:
            split_filename = args.save_diagram_output.split(".")
            pathwithoutextension = ".".join(split_filename[:-1])
            diagram_filename = pathwithoutextension + "_" + suffix + ".csv"
        else:
            diagram_filename = None

    valid_image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.pdf']
    if not any(filename.lower().endswith(fmt) for fmt in valid_image_formats):
        print(filename)
        raise ValueError("Invalid file format. Please provide a valid image format.")

    reliability, confidence, bin_edge, bin_count = reliability_diagram(
        y_true=labels,
        y_proba=probs,
        num_bins=args.plot_bins,
        class_to_plot=args.class_to_calculate,
        save_path=diagram_filename,
    )
    plot_reliability_diagram(
        reliability,
        confidence,
        bin_count,
        save_path=filename,
        title=suffix,
        error_bar=True,
    )
    print("Plot saved to", filename)


def main():
    """
    Main function to parse arguments and perform calibration calculations.
    """
    parser = argparse.ArgumentParser(
        description="Calculate calibration metrics and visualize reliability diagram."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path to the input CSV file. (If there is header,it must be in: "
        "proba_0,proba_1,...,subgroup_1(optional),subgroup_2(optional),...label. "
        "If no header, then the columns must be in the order of "
        "proba_0,proba_1,...,label)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of specific metrics to calculate "
        "(SpiegelhalterZ,ECE-H,MCE-H,HL-H,ECE-C,MCE-C,HL-C,COX,Loess,all). "
        "Default: all",
    )
    parser.add_argument(
        "--prevalence_adjustment",
        default=False,
        action="store_true",
        help="Perform prevalence adjustment (default: False)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap samples (default: 0)",
    )
    parser.add_argument(
        "--bootstrap_ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence interval (default: 0.95)",
    )
    parser.add_argument(
        "--class_to_calculate",
        type=int,
        default=1,
        help="Class to calculate metrics for (default: 1)",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of bins for ECE/MCE/HL calculations (default: 10)",
    )
    parser.add_argument(
        "--topclass",
        default=False,
        action="store_true",
        help="Whether to transform the problem to top-class problem.",
    )
    parser.add_argument(
        "--save_metrics", type=str, help="Save the metrics to a csv file"
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Plot reliability diagram (default: False)",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagram",
    )
    parser.add_argument(
        "--save_plot",
        default="",
        type=str,
        help="Save the plot to a file. Must end with valid image formats.",
    )
    parser.add_argument(
        "--save_diagram_output",
        default="",
        type=str,
        help="Save the reliability diagram output to a file",
    )
    parser.add_argument(
        "--verbose", default=True, action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    loader = data_loader(args.csv_file)

    if args.topclass:
        loader = loader.transform_topclass()

    if not loader.have_subgroup:
        perform_calculation(
            probs=loader.probs, labels=loader.labels, args=args, suffix=""
        )
    else:
        perform_calculation(
            probs=loader.probs, labels=loader.labels, args=args, suffix=""
        )
        for i, subgroup_column in enumerate(loader.subgroup_indices):
            for j, subgroup_class in enumerate(loader.subgroups_class[i]):
                proba = loader.probs[loader.subgroups_index[i][j], :]
                label = loader.labels[loader.subgroups_index[i][j]]
                perform_calculation(
                    probs=proba,
                    labels=label,
                    args=args,
                    suffix=f"subgroup_{i+1}_group_{subgroup_class}",
                )


if __name__ == "__main__":
    main()
