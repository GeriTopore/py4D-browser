from PyQt5.QtWidgets import QDialog, QTabWidget, QVBoxLayout, QWidget

from .bragg_disk_tab import BraggDiskTab
from .probe_kernel_tab import ProbeKernelTab

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class DiskDetectionPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.disk_detection"

    uses_single_action = True
    display_name = "Disk Detection..."

    def __init__(self, parent: "DataViewer", plugin_action, **kwargs):
        super().__init__()

        self.parent = parent
        self.window = None

        plugin_action.triggered.connect(self.launch_window)

    def close(self):
        if self.window is not None:
            self.window.close()

    def launch_window(self):
        if self.window is None:
            self.window = DiskDetectionWindow(parent=self.parent)

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


class DiskDetectionWindow(QDialog):
    def __init__(self, parent: "DataViewer"):
        super().__init__(parent=parent)

        self.parent = parent
        self.probe = None  # the accepted Probe object, once the user hits Accept
        self.probe_radius = None  # measured bright-field disk radius, in pixels

        self.setWindowTitle("Disk Detection")

        self.probe_kernel_tab = ProbeKernelTab(window=self)
        self.bragg_disk_tab = BraggDiskTab(window=self)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.probe_kernel_tab, "Probe Kernel")
        self.tab_widget.addTab(self.bragg_disk_tab, "Bragg Disk Detection")
        self.tab_widget.setTabEnabled(
            self.tab_widget.indexOf(self.bragg_disk_tab), False
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.tab_widget)

        self.resize(1100, 750)

    def probe_accepted(self, probe, probe_radius):
        self.probe = probe
        self.probe_radius = probe_radius

        bragg_tab_index = self.tab_widget.indexOf(self.bragg_disk_tab)
        self.tab_widget.setTabEnabled(bragg_tab_index, True)
        self.bragg_disk_tab.on_probe_accepted()
        self.tab_widget.setCurrentIndex(bragg_tab_index)
