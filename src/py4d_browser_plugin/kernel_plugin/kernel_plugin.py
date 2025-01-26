import py4DSTEM
import h5py
from py4DSTEM import DataCube, data
import os
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QWidget,
    QDoubleSpinBox,
    QFormLayout,
)
from py4D_browser.utils import make_detector, StatusBarWriter
from py4D_browser.menu_actions import show_file_dialog, get_ND, find_calibrations
from py4D_browser.dialogs import ResizeDialog

class KernelPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.kernel"

    ######## optional flags ########
    display_name = "Kernel"

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = True

    # If the plugin only needs a single action button, the browser can opt
    # to have that menu item created automatically under Plugins>[Display Name]
    # and its QAction object passed as `plugin_action`
    # uses_single_action = False

    def __init__(self, parent, plugin_menu, **kwargs):
        super().__init__()
        self.parent = parent

        probe_vacuum = plugin_menu.addAction("Generate probe from vacuum file...")
        probe_vacuum.triggered.connect(self.load_vacuum_file)
        probe_selection = plugin_menu.addAction("Generate probe from selection...")

    def close(self):
        pass  # perform any shutdown activities

    def launch_probekernel_window(self):
        dialog = ProbeKernelDialog(parent=self.parent, vac_datacube=self.vacuum_datacube)
        dialog.show()

    def load_vacuum_file(self, checked=False, mmap=False, binning=1):

        filepath = show_file_dialog(self)

        print(f"Loading file {filepath}")
        extension = os.path.splitext(filepath)[-1].lower()
        print(f"Type: {extension}")
        if extension in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
            file = h5py.File(filepath, "r")
            datacubes = get_ND(file)
            print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
            if len(datacubes) >= 1:
                # Read the first datacube in the HDF5 file into RAM
                print(f"Reading dataset at location {datacubes[0].name}")
                self.vacuum_datacube = py4DSTEM.DataCube(
                    datacubes[0] if mmap else datacubes[0][()]
                )

                R_size, R_units, Q_size, Q_units = find_calibrations(datacubes[0])

                self.vacuum_datacube.calibration.set_R_pixel_size(R_size)
                self.vacuum_datacube.calibration.set_R_pixel_units(R_units)
                self.vacuum_datacube.calibration.set_Q_pixel_size(Q_size)
                self.vacuum_datacube.calibration.set_Q_pixel_units(Q_units)

            else:
                # if no 4D data was found, look for 3D data
                datacubes = get_ND(file, N=3)
                print(f"Found {len(datacubes)} 3D datasets inside the HDF5 file...")
                if len(datacubes) >= 1:
                    array = datacubes[0] if mmap else datacubes[0][()]
                    new_shape = ResizeDialog.get_new_size([1, array.shape[0]], parent=self)
                    self.vacuum_datacube = py4DSTEM.DataCube(
                        array.reshape(*new_shape, *array.shape[1:])
                    )
                else:
                    raise ValueError("No 4D (or even 3D) data detected in the H5 file!")
        elif extension in [".npy"]:
            self.vacuum_datacube = py4DSTEM.DataCube(np.load(filepath))
        else:
            self.vacuum_datacube = py4DSTEM.import_file(
                filepath,
                mem="MEMMAP" if mmap else "RAM",
                binfactor=binning,
            )

        self.launch_probekernel_window()
        # self.generate_probe()

class ProbeKernelDialog(QDialog):
    def __init__(self, parent, vac_datacube):
        super().__init__(parent)
        self.parent = parent

        self.probe = vac_datacube.get_vacuum_probe()

        self.setWindowTitle("Probe from vacuum file")  # Set the window title
        self.setModal(True)  # Make the dialog modal
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.probe_image = pg.ImageView()
        self.probe_image.setImage(self.probe.data)
        layout.addWidget(self.probe_image)

        self.kernel_image = pg.ImageView()
        self.kernel_image.setImage(np.zeros((512, 512)))

        self.probe.get_kernel(mode='flat')

        kernel = self.probe.kernel

        # shift zero frequency to the center
        im_kernel = np.fft.fftshift(kernel)

        self.kernel_image.setImage(im_kernel)
        layout.addWidget(self.kernel_image)

        self.probe_kernel_settings = QGroupBox('Kernel Settings')
        self.kernel_mode = QComboBox()

        self.kernel_mode.addItem("flat")
        self.kernel_mode.addItem("gaussian")
        self.kernel_mode.addItem("sigmoid")
        self.kernel_mode.addItem("sigmoid_log")

        self.sigma = QDoubleSpinBox()
        self.r_outer = QDoubleSpinBox()
        self.r_inner = QDoubleSpinBox()
        self.sigma.setEnabled(False)
        self.r_outer.setEnabled(False)
        self.r_inner.setEnabled(False)

        self.kernel_mode.currentTextChanged.connect(self.update_kernel_settings)

        self.divider = QFrame()
        self.divider.setFrameShape(QFrame.HLine)
        self.divider.setFrameShadow(QFrame.Sunken)
        self.divider.setMinimumHeight(2)

        self.generate_kernel_button = QPushButton("Generate Kernel")

        self.probe_kernel_settings_layout = QFormLayout()
        self.probe_kernel_settings_layout.addRow("Kernel Mode", self.kernel_mode)
        self.probe_kernel_settings_layout.addRow("Sigma", self.sigma)
        self.probe_kernel_settings_layout.addRow("Outer Radius", self.r_outer)
        self.probe_kernel_settings_layout.addRow("Inner Radius", self.r_inner)
        self.probe_kernel_settings_layout.addRow(self.divider)
        self.probe_kernel_settings_layout.addRow(self.generate_kernel_button)

        self.probe_kernel_settings.setLayout(self.probe_kernel_settings_layout)

        layout.addWidget(self.probe_kernel_settings)

    def update_kernel_settings(self, value):
        if value == "sigmoid" or value == "sigmoid_log":
            self.r_outer.setEnabled(True)
            self.r_inner.setEnabled(True)
            self.sigma.setEnabled(False)

        elif value == "gaussian":
            self.r_outer.setEnabled(False)
            self.r_inner.setEnabled(False)
            self.sigma.setEnabled(True)

        else:
            self.r_outer.setEnabled(False)
            self.r_inner.setEnabled(False)
            self.sigma.setEnabled(False)
