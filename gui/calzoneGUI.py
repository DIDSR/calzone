"""

Creating the exe for calzone GUI

conda create -n calzone-gui # new clean environment

conda activate it
install pip
install all the dependencies with pip
install pyinstaller
install nicegui
install pywin32

nicegui-pack --onefile --name "Calzone" calzone_demo.py

check for .exe file in /dist

see: https://nicegui.io/documentation/section_configuration_deployment#package_for_installation

"""

import io
import os
from nicegui import ui, app, events
from typing import Optional
from pathlib import Path
import platform

#from local_file_picker import local_file_picker

from nicegui import native

from argparse import Namespace
import argparse
import numpy as np
from calzone.metrics import CalibrationMetrics, get_CI
from calzone.utils import *
from calzone.vis import plot_reliability_diagram
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("svg")

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
    if args.hl_test_validation:
        df = args.num_bins
    else:
        df = args.num_bins - 2
    metrics_to_calculate = args.metrics.split(",") if args.metrics else ["all"]
    if metrics_to_calculate == ["all"]:
        metrics_to_calculate = "all"
    result = cal_metrics.calculate_metrics(
        y_true=labels,
        y_proba=probs,
        metrics=metrics_to_calculate,
        perform_pervalance_adjustment=args.prevalence_adjustment,
        df = df
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
            df = df
        )
        CI = get_CI(bootstrap_results)
        result = np.vstack((result, np.array(list(CI.values())).T))

    if args.verbose:
        print_metrics(result, keys, args.n_bootstrap, suffix)

    if args.save_metrics:
        save_metrics_to_csv(result, keys, args.save_metrics, suffix)

    if args.plot:
        fig = plot_reliability(labels, probs, args, suffix)
    else:
        fig = None
    return result, fig


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
        if args.save_plot is not None:
            split_filename = args.save_plot.split(".")
            pathwithoutextension = ".".join(split_filename[:-1])
            filename = pathwithoutextension + "_" + suffix + "." + split_filename[-1]
        else:
            filename = None
        if args.save_diagram_output:
            split_filename = args.save_diagram_output.split(".")
            pathwithoutextension = ".".join(split_filename[:-1])
            diagram_filename = pathwithoutextension + "_" + suffix + ".csv"
        else:
            diagram_filename = None
    if args.save_plot is not None:
        valid_image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.pdf']
        if not any(filename.lower().endswith(fmt) for fmt in valid_image_formats):
            print(filename)
            raise ValueError("Invalid file format. Please provide a valid image format.")

    reliability, confidence, bin_edge, bin_count = reliability_diagram(
        y_true=labels,
        y_proba=probs,
        num_bins=args.plot_bins,
        is_equal_freq=args.equal_count,
        class_to_plot=args.class_to_calculate,
        save_path=diagram_filename,
    )
    fig = plot_reliability_diagram(
        reliability,
        confidence,
        bin_count,
        save_path=filename,
        title=suffix,
        error_bar=True,
        return_fig=True
    )
    if args.save_plot is not None:
        print("Plot saved to", filename)
    return fig

def plot_hist_roc(labels, probs, args):
    if args.class_to_calculate is None:
        class_to_calculate = 1
    else:
        class_to_calculate = args.class_to_calculate
    # Calculate ROC curve
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = labels.flatten()
    # Plot histogram
    pos_scores = probs[labels == class_to_calculate, class_to_calculate]
    neg_scores = probs[labels != class_to_calculate, class_to_calculate]
    ax1.hist(pos_scores, bins=20, alpha=0.5, label='Class-of-interest', density=False)
    ax1.hist(neg_scores, bins=20, alpha=0.5, label='Rest', density=False)
    ax1.set_xlabel('Prediction score')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.set_title('Score distributions')
    
    # Plot ROC curve
    fpr, tpr, roc_auc = make_roc_curve(labels, probs, class_to_plot=class_to_calculate)
    ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')

    # Add threshold points
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for thresh in thresholds:
        # Find closest threshold point
        pred_labels = (probs[:, class_to_calculate] >= thresh).astype(int)
        fp = np.sum((pred_labels == 1) & (labels != class_to_calculate))
        tp = np.sum((pred_labels == 1) & (labels == class_to_calculate))
        fpr_point = fp / np.sum(labels != class_to_calculate)
        tpr_point = tp / np.sum(labels == class_to_calculate)
        # Plot point and label
        ax2.plot(fpr_point, tpr_point, 'rx', markersize=6)
        ax2.annotate(f'p={thresh}', (fpr_point, tpr_point), xytext=(6, 6), 
                    textcoords='offset points')

    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate') 
    ax2.legend()
    ax2.set_title('ROC Curve')
    
    plt.tight_layout()
    return fig


def run_calibration(args):
    loader = data_loader(args.csv_file)

    if args.topclass:
        loader = loader.transform_topclass()

    if not loader.have_subgroup:
        _ ,fig = perform_calculation(
            probs=loader.probs, labels=loader.labels, args=args, suffix=""
        )
        if args.plot_hist_roc:
            fig2 = plot_hist_roc(loader.labels, loader.probs, args)
        else:
            fig2 = None
    else:
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

        _ ,fig = perform_calculation(
            probs=loader.probs, labels=loader.labels, args=args, suffix=""
        )
        if args.plot_hist_roc:
            fig2 = plot_hist_roc(label, proba, args)
        else:
            fig2 = None
    return 1, fig, fig2


class local_file_picker(ui.dialog):

    def __init__(self, directory: str, *,
                 upper_limit: Optional[str] = ..., multiple: bool = False, show_hidden_files: bool = False) -> None:
        """Local File Picker

        This is a simple file picker that allows you to select a file from the local filesystem where NiceGUI is running.

        :param directory: The directory to start in.
        :param upper_limit: The directory to stop at (None: no limit, default: same as the starting directory).
        :param multiple: Whether to allow multiple files to be selected.
        :param show_hidden_files: Whether to show hidden files.
        """
        super().__init__()

        self.path = Path(directory).expanduser()
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(directory if upper_limit == ... else upper_limit).expanduser()
        self.show_hidden_files = show_hidden_files

        with self, ui.card():
            self.add_drives_toggle()
            self.grid = ui.aggrid({
                'columnDefs': [{'field': 'name', 'headerName': 'File'}],
                'rowSelection': 'multiple' if multiple else 'single',
            }, html_columns=[0]).classes('w-96').on('cellDoubleClicked', self.handle_double_click)
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=self.close).props('outline')
                #ui.button('Ok', on_click=self._handle_ok)
        self.update_grid()

    def add_drives_toggle(self):
        if platform.system() == 'Windows':
            import win32api
            drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
            self.drives_toggle = ui.toggle(drives, value=drives[0], on_change=self.update_drive)

    def update_drive(self):
        self.path = Path(self.drives_toggle.value).expanduser()
        self.update_grid()

    def update_grid(self) -> None:
        paths = list(self.path.glob('*'))
        if not self.show_hidden_files:
            paths = [p for p in paths if not p.name.startswith('.')]
        paths.sort(key=lambda p: p.name.lower())
        paths.sort(key=lambda p: not p.is_dir())

        self.grid.options['rowData'] = [
            {
                'name': f'📁 <strong>{p.name}</strong>' if p.is_dir() else p.name,
                'path': str(p),
            }
            for p in paths
        ]
        if self.upper_limit is None and self.path != self.path.parent or \
                self.upper_limit is not None and self.path != self.upper_limit:
            self.grid.options['rowData'].insert(0, {
                'name': '📁 <strong>..</strong>',
                'path': str(self.path.parent),
            })
        self.grid.update()

    def handle_double_click(self, e: events.GenericEventArguments) -> None:
        self.path = Path(e.args['data']['path'])
        if self.path.is_dir():
            self.update_grid()
        else:
            self.submit([str(self.path)])

    async def _handle_ok(self):
        rows = await self.grid.get_selected_rows()
        self.submit([r['path'] for r in rows])

def run_program():
    #clear_cache()
    output_area.value = ""  # Clear the output area
    # Get values from UI elements
    csv_file = csv_file_input.value
    save_metrics = save_metrics_input.value
    save_plot = save_plot_input.value

    selected_metrics = [metric for metric, checkbox in metrics_checkboxes.items() 
                    if checkbox.value]

    # Input validation
    if not selected_metrics or not csv_file:
        ui.notify("Error: Please select at least one metric and provide a CSV file.",
                type="error")
        return

    metric_arg = "all" if not selected_metrics else ",".join(selected_metrics)  # Keep as list instead of joining

    if perform_bootstrap_checkbox.value and int(n_bootstrap_input.value) == 0:
        ui.notify("Error: Number of bootstrap samples must be greater than zero when performing bootstrap.",
                type="error")
        return

    # Validate HL test requirements
    if hl_test_validation_checkbox.value:
        if not any(metric in selected_metrics for metric in ["HL-H", "HL-C"]):
            ui.notify("Error: HL test validation requires either HL-H or HL-C metric to be selected.", 
                    type="error")
            return
    

    args = Namespace(
        csv_file=str(csv_file),
        metrics=metric_arg,  # Now a list instead of comma-separated string
        class_to_calculate=int(class_to_calculate_input.value),
        num_bins=int(num_bins_input.value),
        plot_bins=int(plot_bins_input.value),
        save_metrics=str(save_metrics) if save_metrics_input.value else None,
        n_bootstrap=int(n_bootstrap_input.value) if perform_bootstrap_checkbox.value else 0,
        bootstrap_ci=float(bootstrap_ci_input.value) if perform_bootstrap_checkbox.value else None,
        prevalence_adjustment=bool(prevalence_adjustment_checkbox.value),
        plot=bool(plot_checkbox.value),
        equal_count=bool(plot_equal_count_checkbox.value),
        plot_hist_roc=bool(plot_hist_roc_checkbox.value),
        #save_diagram_output=str(save_plot) if plot_checkbox.value else None,
        save_diagram_output=None,
        save_plot=str(save_plot) if save_plot_input.value else None,
        verbose=bool(verbose_checkbox.value),
        topclass=bool(topclass_checkbox.value),
        hl_test_validation=bool(hl_test_validation_checkbox.value)
    )

    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()


    #try:
    # Redirect stdout and stderr to capture all output
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result, fig, fig2 = run_calibration(args)
    
    # Get the captured output
    output = stdout_buffer.getvalue()
    error = stderr_buffer.getvalue()
    
    # Display the captured output in the output area
    output_area.value = output
    if error:
        output_area.value += "\nErrors:\n" + error
    
    # Show success notification if no errors
    if result and not error:
        ui.notify("Calibration metrics calculated successfully", type="positive")
    elif error:
        ui.notify("Completed with errors", type="warning")
    global plot_ui
    global plot_ui2
    if plot_checkbox.value:
        if 'plot_ui' not in globals():
            plot_ui = ui.pyplot(close=False, figsize=(8, 5))
        with plot_ui:
            plot_ui.fig = fig
    # Add display for fig2 when plot_hist_roc is checked
    if plot_hist_roc_checkbox.value and fig2 is not None:
        if 'plot_ui2' not in globals():
            plot_ui2 = ui.pyplot(close=False, figsize=(12, 5))
        with plot_ui2:
            plot_ui2.fig = fig2

def display_plot(plot, fig):
        plot.fig = fig
        plot.update()


def update_csv_file_input(file_path):
    csv_file_input.value = file_path


async def pick_file() -> None:
    result = await local_file_picker('~', multiple=False)
    if result is not None:
        update_csv_file_input(result[0])


app.add_static_files(url_path='/static',local_directory=os.path.dirname(__file__))
with ui.row().classes('w-full justify-center'):
    with ui.column().classes('w-1/3 p-4'):
        with ui.row().classes('items-center gap-4'):
            ui.image("static/logo.png").classes('w-24 h-24')
            ui.label('Calzone GUI (experimental)').classes('text-h4')
        csv_file_input = ui.input(label='CSV File',
                                  placeholder='Enter absolute file path').classes('w-full')
        ui.button('choose file', on_click=pick_file)
        ui.label('Metrics:')
        metrics_options = ["all", "SpiegelhalterZ", "ECE-H", "MCE-H", "HL-H",
                           "ECE-C", "MCE-C", "HL-C", "COX", "Loess"]
        metrics_checkboxes = {metric: ui.checkbox(metric, on_change=lambda m=metric: update_checkboxes(m))
                              for metric in metrics_options}

    with ui.column().classes('w-1/3 p-4'):
        ui.label('Bootstrap:')
        perform_bootstrap_checkbox = ui.checkbox('Perform Bootstrap', value=False)
        n_bootstrap_input = ui.number(label='Number of Bootstrap Samples',
                                      value=0, min=0)
        bootstrap_ci_input = ui.number(label='Bootstrap Confidence Interval',
                                       value=0.95, min=0, max=1, step=0.01)
        ui.label('Prevalence Adjustment:')
        prevalence_adjustment_checkbox = ui.checkbox('Perform Prevalence Adjustment',
                                                     value=False)
        ui.label('Plot:')
        plot_checkbox = ui.checkbox('Plot Reliability Diagram', value=False)
        plot_equal_count_checkbox = ui.checkbox('Use equal-count bin for Reliability Diagram', value=False)
        plot_hist_roc_checkbox = ui.checkbox('Plot Score Histogram and ROC', value=False)
        plot_bins_input = ui.number(label='Number of Bins for Reliability Diagram',
                                    value=10, min=2, step=1)
        save_plot_input = ui.input(label='Save Plot to',
                                   placeholder='Enter file path. Must ends with image extension').classes('w-full')
        ui.label('Setting:')
        class_to_calculate_input = ui.number(label='Class to Calculate Metrics for',
                                             value=1, step=1)
        num_bins_input = ui.number(label='Number of Bins for ECE/MCE/HL Test',
                                   value=10, min=2, step=1)
        hl_test_validation_checkbox = ui.checkbox('HL Test Validation set', value=False)

    with ui.column().classes('w-1/3 p-4'):
        ui.label('Output Paths:')
        save_metrics_input = ui.input(label='Save Metrics to',
                                      placeholder='Enter file path. Must ends with csv.').classes('w-full')
        verbose_checkbox = ui.checkbox('Print Verbose Output', value=True)
        topclass_checkbox = ui.checkbox('Transform to Top-class Problem', value=False)
        ui.button('Run', on_click=run_program).classes('w-full')
with ui.row().classes('w-full justify-center'):
    with ui.column().classes('w-2/3 p-4'):
        output_area = ui.textarea(label='Output').classes('w-full')
        plot_ui = ui.pyplot(close=False, figsize=(8, 5))
        plot_ui2 = ui.pyplot(close=False, figsize=(12, 5))

def update_checkboxes(changed_metric):
    if changed_metric == "all":
        all_checked = metrics_checkboxes["all"].value
        for metric, checkbox in metrics_checkboxes.items():
            if metric != "all":
                checkbox.value = False
                checkbox.disabled = all_checked
    else:
        metrics_checkboxes["all"].value = False
        any_checked = any(checkbox.value for metric, checkbox in metrics_checkboxes.items()
                          if metric != "all")
        metrics_checkboxes["all"].disabled = any_checked

def main():
    ui.run(reload=False, port=native.find_open_port(), native=False)

main()

