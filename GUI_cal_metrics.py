import argparse
import subprocess
from nicegui import ui, app, events
from typing import Optional
from pathlib import Path
import platform


def run_program():
    clear_cache()
    output_area.value = ""  # Clear the output area
    plot_image.clear()
    plot_image.set_source(None)  # Set the image source to None
    ui.update(plot_image)
    csv_file = csv_file_input.value
    save_metrics = save_metrics_input.value
    save_plot = save_plot_input.value

    selected_metrics = [metric for metric, checkbox in metrics_checkboxes.items()
                        if checkbox.value]

    if not selected_metrics or not csv_file:
        ui.notify("Error: Please select at least one metric and provide a CSV file.",
                  type="error")
        return

    metric_arg = "all" if not selected_metrics else ",".join(selected_metrics)

    if perform_bootstrap_checkbox.value and int(n_bootstrap_input.value) == 0:
        ui.notify("Error: Number of bootstrap samples must be greater than zero when performing bootstrap.",
                  type="error")
        return

    args = [
        "--csv_file", str(csv_file),
        "--metrics", str(metric_arg),
        "--class_to_calculate", str(class_to_calculate_input.value),
        "--num_bins", str(num_bins_input.value),
        "--plot_bins", str(plot_bins_input.value),
    ]

    if save_metrics_input.value:
        args.append("--save_metrics")
        args.append(str(save_metrics))

    if perform_bootstrap_checkbox.value:
        args.append("--n_bootstrap")
        args.append(str(int(n_bootstrap_input.value)))
        args.append("--bootstrap_ci")
        args.append(str(bootstrap_ci_input.value))

    if prevalence_adjustment_checkbox.value:
        args.append("--prevalence_adjustment")

    if plot_checkbox.value:
        if not save_plot:
            ui.notify("Error: Please provide a path to save the plot.", type="error")
            return
        args.append("--plot")
        args.append("--save_plot")
        args.append(str(save_plot))
    if verbose_checkbox.value:
        args.append("--verbose")
    if topclass_checkbox.value:
        args.append("--topclass")
    if hl_test_validation_checkbox.value:
        if not any(metric in selected_metrics for metric in ["HL-H", "HL-C"]):
            ui.notify("Error: HL test validation requires either HL-H or HL-C metric to be selected.", type="error")
            return
        args.append("--hl_test_validation")

    command = ["cal_metrics"] + args
    print("Running command:", " ".join(command))

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
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

    plot_image.clear()
    plot_image.set_source(None)  # Set the image source to None
    ui.update(plot_image)


def update_csv_file_input(file_path):
    csv_file_input.value = file_path


async def pick_file() -> None:
    result = await local_file_picker('~', multiple=False)
    update_csv_file_input(result[0])


with ui.row().classes('w-full justify-center'):
    with ui.column().classes('w-1/3 p-4'):
        ui.label('calzone GUI').classes('text-h4')
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
        any_checked = any(checkbox.value for metric, checkbox in metrics_checkboxes.items()
                          if metric != "all")
        metrics_checkboxes["all"].disabled = any_checked

def main():
    ui.run()

main()

class local_file_picker(ui.dialog):

    def __init__(self, directory: str, *,
                 upper_limit: Optional[str] = ..., multiple: bool = False,
                 show_hidden_files: bool = False) -> None:
        """
        This local file picker is developed by nicegui author. 
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
                ui.button('Ok', on_click=self._handle_ok)
        self.update_grid()

    def add_drives_toggle(self):
        if platform.system() == 'Windows':
            import win32api
            drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
            self.drives_toggle = ui.toggle(drives, value=drives[0],
                                           on_change=self.update_drive)

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