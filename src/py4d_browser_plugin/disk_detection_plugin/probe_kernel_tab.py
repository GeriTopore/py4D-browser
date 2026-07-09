import os

import h5py
import numpy as np
import pyqtgraph as pg
import py4DSTEM
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.dialogs import ResizeDialog
from py4D_browser.menu_actions import find_calibrations, get_ND
from py4D_browser.utils import DetectorInfo, DetectorShape

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer
    from .disk_detection_plugin import DiskDetectionWindow


def _vacuum_probe_from_single_pixel(datacube, xc, yc, threshold, expansion, opening):
    """
    Work around a py4DSTEM bug (as of 0.14.18): DataCube.get_vacuum_probe's
    internal averaging loop is skipped entirely when the ROI contains
    exactly one scan position, leaving its `probe` variable as a raw view
    into datacube.data (sometimes read-only, and the wrong dtype for the
    in-place `probe *= mask` that follows) -- both fail. Replicate the same
    thresholding/expansion/opening masking on a writable float copy
    instead, since averaging over a single pattern is just that pattern.
    """
    from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt

    probe = datacube.data[xc, yc, :, :].astype(np.float64)
    mask = probe > np.max(probe) * threshold
    mask = binary_opening(mask, iterations=opening)
    mask = binary_dilation(mask, iterations=1)
    mask = (
        np.cos(
            (np.pi / 2)
            * np.minimum(distance_transform_edt(np.logical_not(mask)) / expansion, 1)
        )
        ** 2
    )
    probe = py4DSTEM.Probe(probe * mask)
    datacube.attach(probe)
    return probe


def _load_datacube_from_file(
    filepath: str, dialog_parent: QWidget
) -> py4DSTEM.DataCube:
    """
    Load a standalone DataCube from a file on disk, using the same
    format-detection logic as py4D_browser.menu_actions.load_file.
    """
    extension = os.path.splitext(filepath)[-1].lower()

    if extension in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
        file = h5py.File(filepath, "r")
        datacubes = get_ND(file)
        if len(datacubes) >= 1:
            parent_group = "/".join(datacubes[0].name.split("/")[:-1])
            if len(parent_group) > 1 and "emd_group_type" in file[parent_group].attrs:
                return py4DSTEM.DataCube.from_h5(datacubes[0].file[parent_group])

            datacube = py4DSTEM.DataCube(datacubes[0][()])
            R_size, R_units, Q_size, Q_units = find_calibrations(datacubes[0])
            datacube.calibration.set_R_pixel_size(R_size)
            datacube.calibration.set_R_pixel_units(R_units)
            datacube.calibration.set_Q_pixel_size(Q_size)
            datacube.calibration.set_Q_pixel_units(Q_units)
            return datacube

        datacubes = get_ND(file, N=3)
        if len(datacubes) >= 1:
            array = datacubes[0][()]
            new_shape = ResizeDialog.get_new_size(
                [1, array.shape[0]], parent=dialog_parent
            )
            return py4DSTEM.DataCube(array.reshape(*new_shape, *array.shape[1:]))

        raise ValueError("No 4D (or even 3D) data detected in the H5 file!")

    elif extension == ".npy":
        return py4DSTEM.DataCube(np.load(filepath))

    else:
        return py4DSTEM.import_file(filepath, mem="RAM")


class ProbeKernelTab(QWidget):
    def __init__(self, window: "DiskDetectionWindow"):
        super().__init__()

        self.window = window

        self.probe = None
        self.alpha = None
        self.qx0 = None
        self.qy0 = None
        self._source = None
        self._vacuum_datacube = None

        # ---- vacuum region source ----
        source_box = QGroupBox("Probe Source")
        source_layout = QVBoxLayout()

        button_row = QHBoxLayout()
        use_selection_button = QPushButton("Use Current Selection")
        use_selection_button.clicked.connect(self.use_current_selection)
        button_row.addWidget(use_selection_button)

        load_vacuum_button = QPushButton("Load Vacuum File...")
        load_vacuum_button.clicked.connect(self.load_vacuum_file)
        button_row.addWidget(load_vacuum_button)

        use_synthetic_button = QPushButton("Synthetic Probe")
        use_synthetic_button.clicked.connect(self.use_synthetic_probe)
        button_row.addWidget(use_synthetic_button)
        source_layout.addLayout(button_row)

        self.source_label = QLabel("No vacuum region selected yet")
        source_layout.addWidget(self.source_label)

        # Only used when the source is "Synthetic Probe". Radius defaults to
        # (and is kept in sync with) the most recently measured bright-field
        # disk radius, so switching to a synthetic probe starts from a
        # physically sensible size.
        synth_form = QFormLayout()
        self.synth_radius_spin = QDoubleSpinBox()
        self.synth_radius_spin.setRange(0.5, 1000.0)
        self.synth_radius_spin.setValue(10.0)
        synth_form.addRow("Synthetic Radius (px)", self.synth_radius_spin)

        self.synth_width_spin = QDoubleSpinBox()
        self.synth_width_spin.setRange(0.1, 100.0)
        self.synth_width_spin.setValue(4.0)
        self.synth_width_spin.setEnabled(False)
        synth_form.addRow("Synthetic Edge Width (px)", self.synth_width_spin)

        source_layout.addLayout(synth_form)
        source_box.setLayout(source_layout)

        # ---- mask settings + probe generation ----
        mask_box = QGroupBox("Mask Settings")
        mask_form = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.0)
        mask_form.addRow("Mask Threshold", self.threshold_spin)

        self.expansion_spin = QSpinBox()
        self.expansion_spin.setRange(0, 500)
        self.expansion_spin.setValue(12)
        mask_form.addRow("Mask Expansion", self.expansion_spin)

        self.opening_spin = QSpinBox()
        self.opening_spin.setRange(0, 500)
        self.opening_spin.setValue(3)
        mask_form.addRow("Mask Opening", self.opening_spin)

        generate_probe_button = QPushButton("Generate Probe")
        generate_probe_button.clicked.connect(self.generate_probe)

        mask_layout = QVBoxLayout()
        mask_layout.addLayout(mask_form)
        mask_layout.addWidget(generate_probe_button)
        mask_box.setLayout(mask_layout)

        self.probe_view = pg.ImageView()
        self.probe_view.setImage(np.zeros((256, 256)))

        # ---- kernel settings ----
        kernel_box = QGroupBox("Kernel Settings")
        kernel_form = QFormLayout()

        self.kernel_mode_combo = QComboBox()
        self.kernel_mode_combo.addItems(["flat", "gaussian", "sigmoid", "sigmoid_log"])
        self.kernel_mode_combo.currentTextChanged.connect(
            self.update_kernel_controls_enabled
        )
        kernel_form.addRow("Kernel Mode", self.kernel_mode_combo)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(4.0)
        kernel_form.addRow("Sigma", self.sigma_spin)

        self.r_inner_spin = QDoubleSpinBox()
        self.r_inner_spin.setRange(0.0, 10.0)
        self.r_inner_spin.setSingleStep(0.1)
        self.r_inner_spin.setValue(1.0)
        kernel_form.addRow("Inner Radius (x probe radius)", self.r_inner_spin)

        self.r_outer_spin = QDoubleSpinBox()
        self.r_outer_spin.setRange(0.0, 10.0)
        self.r_outer_spin.setSingleStep(0.1)
        self.r_outer_spin.setValue(4.0)
        kernel_form.addRow("Outer Radius (x probe radius)", self.r_outer_spin)

        generate_kernel_button = QPushButton("Generate Kernel")
        generate_kernel_button.clicked.connect(self.generate_kernel)

        kernel_layout = QVBoxLayout()
        kernel_layout.addLayout(kernel_form)
        kernel_layout.addWidget(generate_kernel_button)
        kernel_box.setLayout(kernel_layout)

        self.kernel_view = pg.ImageView()
        self.kernel_view.setImage(np.zeros((256, 256)))

        # Line profiles through the kernel center, along x and along y,
        # so the falloff shape (e.g. gaussian/sigmoid rolloff) can be judged.
        self.kernel_profile_plot = pg.PlotWidget()
        self.kernel_profile_plot.addLegend()
        self.kernel_profile_plot.setMaximumHeight(150)
        self._kernel_x_curve = self.kernel_profile_plot.plot(
            pen=pg.mkPen("y", width=2), name="x profile"
        )
        self._kernel_y_curve = self.kernel_profile_plot.plot(
            pen=pg.mkPen("c", width=2), name="y profile"
        )

        self.accept_button = QPushButton("Accept")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self.accept)

        # ---- overall layout ----
        left_layout = QVBoxLayout()
        left_layout.addWidget(source_box)
        left_layout.addWidget(mask_box)
        left_layout.addWidget(kernel_box)
        left_layout.addWidget(self.accept_button)
        left_layout.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Splitters (rather than plain layouts) so the panes can be resized
        # by dragging their borders, same as the main window's own
        # real-space/FFT split.
        kernel_splitter = QSplitter(QtCore.Qt.Vertical)
        kernel_splitter.addWidget(self.kernel_view)
        kernel_splitter.addWidget(self.kernel_profile_plot)
        kernel_splitter.setSizes([500, 150])

        views_splitter = QSplitter(QtCore.Qt.Horizontal)
        views_splitter.addWidget(self.probe_view)
        views_splitter.addWidget(kernel_splitter)

        main_splitter = QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(views_splitter)
        main_splitter.setSizes([300, 900])

        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        self.setLayout(layout)

        self.update_kernel_controls_enabled(self.kernel_mode_combo.currentText())

    def use_current_selection(self):
        self._vacuum_datacube = None
        self._set_source("selection", "Source: selection on current dataset")

    def load_vacuum_file(self):
        parent = self.window.parent
        try:
            filepath = parent.show_file_dialog()
        except ValueError:
            return

        try:
            datacube = _load_datacube_from_file(filepath, dialog_parent=self)
        except Exception as exc:
            parent.statusBar().showMessage(f"Couldn't load vacuum file: {exc}", 5_000)
            return

        self._vacuum_datacube = datacube
        self._set_source("vacuum_file", f"Source: {os.path.basename(filepath)}")

    def use_synthetic_probe(self):
        self._vacuum_datacube = None
        self._set_source("synthetic", "Source: synthetic probe")

    def _set_source(self, source, label_text):
        self._source = source
        self.source_label.setText(label_text)
        self.synth_width_spin.setEnabled(source == "synthetic")

    def _selection_roi_mask(self, parent) -> Optional[np.ndarray]:
        """
        Build a boolean R-space mask from whatever is currently selected on
        the main window's virtual image -- a rectangular region, or a
        single scan position (point selector).
        """
        detector: DetectorInfo = parent.get_virtual_image_detector()

        if detector["shape"] is DetectorShape.RECTANGULAR:
            return detector["mask"]

        if detector["shape"] is DetectorShape.POINT:
            xc, yc = detector["point"]
            mask = np.zeros(parent.datacube.Rshape, dtype=np.bool_)
            mask[xc, yc] = True
            return mask

        parent.statusBar().showMessage(
            "Select a point or rectangular region on the virtual image to "
            "use as the vacuum region.",
            5_000,
        )
        return None

    def _measure_probe_from_current_selection(self, parent):
        """
        Measure a (probe, alpha, qx0, qy0) from whatever is currently
        selected on the main window (point or rectangular), or return None
        (with a status bar message already shown) if there's no usable
        selection.
        """
        mask = self._selection_roi_mask(parent)
        if mask is None:
            return None

        threshold = self.threshold_spin.value()
        expansion = self.expansion_spin.value()
        opening = self.opening_spin.value()

        if mask.sum() == 1:
            xc, yc = (int(i[0]) for i in np.nonzero(mask))
            probe = _vacuum_probe_from_single_pixel(
                parent.datacube, xc, yc, threshold, expansion, opening
            )
        else:
            probe = parent.datacube.get_vacuum_probe(
                threshold=threshold, expansion=expansion, opening=opening, ROI=mask
            )
        alpha, qx0, qy0 = parent.datacube.get_probe_size(probe.probe)
        return probe, alpha, qx0, qy0

    def generate_probe(self):
        parent = self.window.parent
        if parent.datacube is None:
            parent.statusBar().showMessage(
                "Load a dataset in the main window first!", 5_000
            )
            return

        if self._source == "selection":
            result = self._measure_probe_from_current_selection(parent)
            if result is None:
                return
            self.probe, self.alpha, self.qx0, self.qy0 = result
            self.synth_radius_spin.setValue(self.alpha)

        elif self._source == "vacuum_file":
            if self._vacuum_datacube is None:
                parent.statusBar().showMessage("Load a vacuum file first!", 5_000)
                return
            self.probe = self._vacuum_datacube.get_vacuum_probe(
                threshold=self.threshold_spin.value(),
                expansion=self.expansion_spin.value(),
                opening=self.opening_spin.value(),
            )
            self.alpha, self.qx0, self.qy0 = parent.datacube.get_probe_size(
                self.probe.probe
            )
            self.synth_radius_spin.setValue(self.alpha)

        elif self._source == "synthetic":
            # Always re-measure the BF disk radius from the current
            # selection, so the synthetic probe reflects it exactly rather
            # than depending on a manually-entered or possibly-stale value.
            # Falls back to the current spinbox value (e.g. hand-entered)
            # if there's no usable selection to measure from.
            result = self._measure_probe_from_current_selection(parent)
            if result is not None:
                _, measured_alpha, _, _ = result
                self.synth_radius_spin.setValue(measured_alpha)

            Qshape = (parent.datacube.Q_Nx, parent.datacube.Q_Ny)
            radius = self.synth_radius_spin.value()
            self.probe = py4DSTEM.Probe.generate_synthetic_probe(
                radius=radius, width=self.synth_width_spin.value(), Qshape=Qshape
            )
            self.alpha = radius
            self.qx0, self.qy0 = Qshape[0] / 2.0, Qshape[1] / 2.0

        else:
            parent.statusBar().showMessage(
                "Choose a vacuum region source first!", 5_000
            )
            return

        self.probe_view.setImage(self.probe.probe, autoLevels=True, autoRange=True)
        self.accept_button.setEnabled(False)

    def update_kernel_controls_enabled(self, mode):
        self.sigma_spin.setEnabled(mode == "gaussian")
        self.r_inner_spin.setEnabled(mode in ("sigmoid", "sigmoid_log"))
        self.r_outer_spin.setEnabled(mode in ("sigmoid", "sigmoid_log"))

    def generate_kernel(self):
        parent = self.window.parent
        if self.probe is None:
            parent.statusBar().showMessage("Generate a probe first!", 5_000)
            return

        mode = self.kernel_mode_combo.currentText()
        kwargs = {}
        if mode == "gaussian":
            kwargs["sigma"] = self.sigma_spin.value()
        elif mode in ("sigmoid", "sigmoid_log"):
            kwargs["radii"] = (
                self.alpha * self.r_inner_spin.value(),
                self.alpha * self.r_outer_spin.value(),
            )

        self.probe.get_kernel(mode=mode, origin=(self.qx0, self.qy0), **kwargs)
        kernel_shifted = np.fft.fftshift(self.probe.kernel)
        self.kernel_view.setImage(kernel_shifted, autoLevels=True, autoRange=True)

        cx, cy = kernel_shifted.shape[0] // 2, kernel_shifted.shape[1] // 2
        self._kernel_x_curve.setData(
            np.arange(kernel_shifted.shape[0]), kernel_shifted[:, cy]
        )
        self._kernel_y_curve.setData(
            np.arange(kernel_shifted.shape[1]), kernel_shifted[cx, :]
        )

        self.accept_button.setEnabled(True)

    def accept(self):
        self.window.probe_accepted(self.probe, self.alpha)
