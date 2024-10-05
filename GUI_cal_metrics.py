import argparse
import numpy as np
import subprocess
import time
import matplotlib.pyplot as plt
from nicegui import ui, app


def run_program():
    csv_file = csv_file_input.value
    save_metrics = save_metrics_input.value
    save_plot = save_plot_input.value

    selected_metrics = [metric for metric, checkbox in metrics_checkboxes.items() if checkbox.value]
    metric_arg = "all" if not selected_metrics else ",".join(selected_metrics)

    args = [
        "--csv_file", str(csv_file),
        "--metrics", str(metric_arg),
        "--n_bootstrap", str(int(n_bootstrap_input.value)),
        "--bootstrap_ci", str(bootstrap_ci_input.value),
        "--class_to_calculate", str(class_to_calculate_input.value),
        "--num_bins", str(num_bins_input.value),
        "--save_metrics", str(save_metrics),
        "--plot_bins", str(plot_bins_input.value),
    ]

    if prevalence_adjustment_checkbox.value:
        args.append("--prevalence_adjustment")

    if plot_checkbox.value:
        args.append("--plot")
        args.append("--save_plot")
        args.append(str(save_plot))
    if verbose_checkbox.value:
        args.append("--verbose")
    if topclass_checkbox.value:
        args.append("--topclass")

    command = ["python", "cal_metrics.py"] + args
    print("Running command:", " ".join(command))
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()

    output_area.value = output
    if error:
        output_area.value += "\nErrors:\n" + error

    if plot_checkbox.value:
        display_plot(save_plot)

def display_plot(plot_path):
    plot_image.clear()
    plot_image.set_source(plot_path)
    ui.update(plot_image)

def clear_cache():
    ui.run_javascript('''
        function clearCache() {
            if ('caches' in window) {
                caches.keys().then(function(names) {
                    for (let name of names)
                        caches.delete(name);
                });
            }
            
            if (window.performance && window.performance.memory) {
                window.performance.memory.usedJSHeapSize = 0;
            }
            
            sessionStorage.clear();
            localStorage.clear();
            
            document.cookie.split(";").forEach(function(c) { 
                document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
            });
            
            location.reload(true);
        }
        clearCache();
    ''')
    ui.notify('Browser cache cleared')
    
    plot_image.clear()
    ui.update(plot_image)

with ui.row().classes('w-full justify-center'):
    with ui.column().classes('w-1/3 p-4'):
        ui.label('Calibration Metrics GUI').classes('text-h4')
        csv_file_input = ui.input(label='CSV File', placeholder='Enter file path').classes('w-full')
        ui.label('Metrics:')
        metrics_options = ["all", "SpiegelhalterZ", "ECE-H", "MCE-H", "HL-H", "ECE-C", "MCE-C", "HL-C", "COX", "Loess"]
        metrics_checkboxes = {metric: ui.checkbox(metric, on_change=lambda m=metric: update_checkboxes(m)) for metric in metrics_options}
        

    with ui.column().classes('w-1/3 p-4'):
        ui.label('Bootstrap:')
        n_bootstrap_input = ui.number(label='Number of Bootstrap Samples', value=0, min=0)
        bootstrap_ci_input = ui.number(label='Bootstrap Confidence Interval', value=0.95, min=0, max=1, step=0.01)
        ui.label('Setting:')
        class_to_calculate_input = ui.number(label='Class to Calculate Metrics for', value=1, step=1)
        num_bins_input = ui.number(label='Number of Bins for ECE/MCE/HL Test', value=10, min=2, step=1)
        save_metrics_input = ui.input(label='Save Metrics to', placeholder='Enter file path').classes('w-full')

    with ui.column().classes('w-1/3 p-4'):
        ui.label('Prevalence Adjustment:')
        prevalence_adjustment_checkbox = ui.checkbox('Perform Prevalence Adjustment', value=False)
        ui.label('Plot:')
        plot_checkbox = ui.checkbox('Plot Reliability Diagram', value=False)
        plot_bins_input = ui.number(label='Number of Bins for Reliability Diagram', value=10, min=2, step=1)
        save_plot_input = ui.input(label='Save Plot to', placeholder='Enter file path').classes('w-full')
        verbose_checkbox = ui.checkbox('Print Verbose Output', value=True)
        topclass_checkbox = ui.checkbox('Transform to Top-class Problem', value=False)
        ui.button('Run', on_click=run_program).classes('w-full')
        ui.button('Clear Browser Cache', on_click=clear_cache).classes('w-full')

with ui.row().classes('w-full justify-center'):
    with ui.column().classes('w-2/3 p-4'):
        output_area = ui.textarea(label='Output').classes('w-full')
    with ui.column().classes('w-1/3 p-4'):
        plot_image = ui.image().classes('w-full')

def update_checkboxes(changed_metric):
    if changed_metric == "all":
        all_checked = metrics_checkboxes["all"].value
        for metric, checkbox in metrics_checkboxes.items():
            if metric != "all":
                checkbox.value = False
                checkbox.disabled = all_checked
    else:
        metrics_checkboxes["all"].value = False
        any_checked = any(checkbox.value for metric, checkbox in metrics_checkboxes.items() if metric != "all")
        metrics_checkboxes["all"].disabled = any_checked

ui.run()