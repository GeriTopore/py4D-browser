import os
from functools import partial

import h5py
import numpy as np
import pyqtgraph as pg
import py4DSTEM
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.dialogs import ResizeDialog
from py4D_browser.menu_actions import find_calibrations, get_ND
from py4D_browser.utils import DetectorInfo, DetectorShape, pg_point_roi

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer


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
        self.synth_radius_spin.setEnabled(False)
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

        kernel_views_layout = QVBoxLayout()
        kernel_views_layout.addWidget(self.kernel_view)
        kernel_views_layout.addWidget(self.kernel_profile_plot)
        kernel_views_widget = QWidget()
        kernel_views_widget.setLayout(kernel_views_layout)

        views_layout = QHBoxLayout()
        views_layout.addWidget(self.probe_view)
        views_layout.addWidget(kernel_views_widget)
        views_widget = QWidget()
        views_widget.setLayout(views_layout)

        layout = QHBoxLayout()
        layout.addWidget(left_widget, 1)
        layout.addWidget(views_widget, 3)
        self.setLayout(layout)

        self.update_kernel_controls_enabled(self.kernel_mode_combo.currentText())

    def use_current_selection(self):
        self._vacuum_datacube = None
        self._set_source(
            "selection", "Source: rectangular selection on current dataset"
        )

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

        is_synthetic = source == "synthetic"
        self.threshold_spin.setEnabled(not is_synthetic)
        self.expansion_spin.setEnabled(not is_synthetic)
        self.opening_spin.setEnabled(not is_synthetic)
        self.synth_radius_spin.setEnabled(is_synthetic)
        self.synth_width_spin.setEnabled(is_synthetic)

    def generate_probe(self):
        parent = self.window.parent
        if parent.datacube is None:
            parent.statusBar().showMessage(
                "Load a dataset in the main window first!", 5_000
            )
            return

        if self._source == "selection":
            detector: DetectorInfo = parent.get_virtual_image_detector()
            if detector["shape"] is not DetectorShape.RECTANGULAR:
                parent.statusBar().showMessage(
                    "Select a rectangular region on the virtual image to use as "
                    "the vacuum region.",
                    5_000,
                )
                return
            self.probe = parent.datacube.get_vacuum_probe(
                threshold=self.threshold_spin.value(),
                expansion=self.expansion_spin.value(),
                opening=self.opening_spin.value(),
                ROI=detector["mask"],
            )
            self.alpha, self.qx0, self.qy0 = parent.datacube.get_probe_size(
                self.probe.probe
            )
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


class BraggDiskSettingsPane(QGroupBox):
    def __init__(self):
        super().__init__("Detection Parameters")

        form = QFormLayout()

        self.corr_power_spin = QDoubleSpinBox()
        self.corr_power_spin.setRange(0.0, 1.0)
        self.corr_power_spin.setSingleStep(0.02)
        self.corr_power_spin.setValue(1.0)
        form.addRow("Correlation Power", self.corr_power_spin)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(0.0)
        form.addRow("Sigma (0 = off)", self.sigma_spin)

        self.sigma_cc_spin = QDoubleSpinBox()
        self.sigma_cc_spin.setRange(0.0, 100.0)
        self.sigma_cc_spin.setValue(0.0)
        form.addRow("Correlation Smoothing Sigma", self.sigma_cc_spin)

        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["pixel", "poly", "multicorr"])
        self.subpixel_combo.setCurrentText("poly")
        form.addRow("Subpixel Mode", self.subpixel_combo)

        self.upsample_factor_spin = QSpinBox()
        self.upsample_factor_spin.setRange(0, 256)
        self.upsample_factor_spin.setValue(0)
        form.addRow("Upsample Factor", self.upsample_factor_spin)

        self.min_abs_intensity_spin = QDoubleSpinBox()
        self.min_abs_intensity_spin.setRange(0.0, 1e6)
        self.min_abs_intensity_spin.setDecimals(4)
        self.min_abs_intensity_spin.setValue(0.0)
        form.addRow("Minimum Absolute Intensity", self.min_abs_intensity_spin)

        self.min_rel_intensity_spin = QDoubleSpinBox()
        self.min_rel_intensity_spin.setRange(0.0, 1.0)
        self.min_rel_intensity_spin.setDecimals(5)
        self.min_rel_intensity_spin.setSingleStep(0.001)
        self.min_rel_intensity_spin.setValue(0.0)
        form.addRow("Minimum Relative Intensity", self.min_rel_intensity_spin)

        self.relative_to_peak_spin = QSpinBox()
        self.relative_to_peak_spin.setRange(0, 20)
        self.relative_to_peak_spin.setValue(0)
        form.addRow("Relative to Peak #", self.relative_to_peak_spin)

        self.min_peak_spacing_spin = QSpinBox()
        self.min_peak_spacing_spin.setRange(0, 1000)
        self.min_peak_spacing_spin.setValue(0)
        form.addRow("Minimum Peak Spacing (px)", self.min_peak_spacing_spin)

        self.edge_boundary_spin = QSpinBox()
        self.edge_boundary_spin.setRange(0, 1000)
        self.edge_boundary_spin.setValue(0)
        form.addRow("Edge Boundary (px)", self.edge_boundary_spin)

        self.max_num_peaks_spin = QSpinBox()
        self.max_num_peaks_spin.setRange(0, 1000)
        self.max_num_peaks_spin.setValue(0)
        form.addRow("Max Number of Peaks", self.max_num_peaks_spin)

        self.cuda_checkbox = QCheckBox()
        form.addRow("Use CUDA", self.cuda_checkbox)

        self.setLayout(form)

        self._all_spinboxes = [
            self.corr_power_spin,
            self.sigma_spin,
            self.sigma_cc_spin,
            self.upsample_factor_spin,
            self.min_abs_intensity_spin,
            self.min_rel_intensity_spin,
            self.relative_to_peak_spin,
            self.min_peak_spacing_spin,
            self.edge_boundary_spin,
            self.max_num_peaks_spin,
        ]

    def connect_changed(self, slot):
        for spin in self._all_spinboxes:
            spin.valueChanged.connect(slot)
        self.subpixel_combo.currentTextChanged.connect(slot)
        self.cuda_checkbox.stateChanged.connect(slot)

    def get_params(self):
        return dict(
            corrPower=self.corr_power_spin.value(),
            sigma=self.sigma_spin.value() or None,
            sigma_cc=self.sigma_cc_spin.value(),
            subpixel=self.subpixel_combo.currentText(),
            upsample_factor=self.upsample_factor_spin.value(),
            minAbsoluteIntensity=self.min_abs_intensity_spin.value(),
            minRelativeIntensity=self.min_rel_intensity_spin.value(),
            relativeToPeak=self.relative_to_peak_spin.value(),
            minPeakSpacing=self.min_peak_spacing_spin.value(),
            edgeBoundary=self.edge_boundary_spin.value(),
            maxNumPeaks=self.max_num_peaks_spin.value(),
            CUDA=self.cuda_checkbox.isChecked(),
        )


class BraggPreviewPane(QGroupBox):
    def __init__(self, title):
        super().__init__(title)

        self.last_dp = None
        self._has_shown_dp = False

        self.rs_view = pg.ImageView()
        self.rs_view.setImage(np.zeros((25, 25)))
        self.rs_selector = pg_point_roi(self.rs_view.getView())

        self.dp_view = pg.ImageView()
        self.dp_view.setImage(np.zeros((512, 512)))

        # Bragg disk positions detected at this pane's scan position, drawn
        # as open rings sized to match the measured bright-field disk
        # diameter (set via set_marker_diameter once a probe is accepted).
        self.scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("g", width=2),
            brush=None,
            symbol="o",
            pxMode=False,  # size is in data (pixel) units, not screen pixels,
            # so markers scale with the image the same way the disks do
        )
        self.dp_view.getView().addItem(self.scatter)

        layout = QVBoxLayout()
        layout.addWidget(self.rs_view)
        layout.addWidget(self.dp_view)
        self.setLayout(layout)

    def set_realspace_image(self, image: Optional[np.ndarray]):
        if image is not None:
            self.rs_view.setImage(image, autoLevels=True, autoRange=True)

    def set_marker_diameter(self, diameter):
        self.scatter.setSize(diameter)

    def get_scan_position(self, datacube):
        roi_state = self.rs_selector.saveState()
        y0, x0 = roi_state["pos"]
        xc, yc = int(x0 + 1), int(y0 + 1)
        xc = int(np.clip(xc, 0, datacube.R_Nx - 1))
        yc = int(np.clip(yc, 0, datacube.R_Ny - 1))
        return xc, yc

    def update_dp(self, dp, scale_fn, relevel):
        self.last_dp = dp
        scaled = scale_fn(dp)

        if relevel or not self._has_shown_dp:
            levels = tuple(np.percentile(scaled, [0.1, 99.9]))
            self.dp_view.setImage(
                scaled,
                autoLevels=False,
                levels=levels,
                autoRange=not self._has_shown_dp,
            )
            self._has_shown_dp = True
        else:
            # Leave levels untouched -- pyqtgraph does not reset them when
            # autoLevels=False and no explicit levels are given, so any
            # level range the user dragged in by hand on the histogram
            # widget survives a scan-position-only update.
            self.dp_view.setImage(scaled, autoLevels=False, autoRange=False)

    def update_scatter(self, qx, qy):
        # ScatterPlotItem positions map directly onto the array indices
        # ImageView.setImage was given -- no axis swap needed here (verified
        # empirically; the old interactive_disk_detection branch's swap was
        # a leftover from a different image-display setup and is wrong here).
        spots = [{"pos": [x, y], "data": 1} for x, y in zip(qx, qy)]
        self.scatter.setData(spots)


class BraggDiskTab(QWidget):
    # Start with a single preview pane; this caps how many more the user
    # can add via the "Add Preview Position" button.
    MAX_ADDITIONAL_PANES = 3

    def __init__(self, window: "DiskDetectionWindow"):
        super().__init__()

        self.window = window
        self.panes = []

        self.settings_pane = BraggDiskSettingsPane()
        self.settings_pane.connect_changed(self.on_detection_params_changed)

        scaling_box = self._build_scaling_box()

        self.find_all_button = QPushButton("Find All Bragg Disks")
        self.find_all_button.clicked.connect(self.find_all)

        self.add_pane_button = QPushButton("Add Preview Position")
        self.add_pane_button.clicked.connect(self.add_pane)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.settings_pane)
        left_layout.addWidget(scaling_box)
        left_layout.addWidget(self.find_all_button)
        left_layout.addWidget(self.add_pane_button)
        left_layout.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self.previews_layout = QHBoxLayout()
        previews_widget = QWidget()
        previews_widget.setLayout(self.previews_layout)

        layout = QHBoxLayout()
        layout.addWidget(left_widget, 1)
        layout.addWidget(previews_widget, 4)
        self.setLayout(layout)

        self._append_pane()

    def _build_scaling_box(self):
        scaling_box = QGroupBox("Diffraction Display Scaling")
        scaling_layout = QVBoxLayout()

        self.scaling_group = QButtonGroup(self)
        self.linear_radio = QRadioButton("Linear")
        self.log_radio = QRadioButton("Log")
        self.power_radio = QRadioButton("Power")
        self.power_radio.setChecked(True)
        for button in (self.linear_radio, self.log_radio, self.power_radio):
            self.scaling_group.addButton(button)
            scaling_layout.addWidget(button)
            # only redraw once, when a button becomes checked -- QButtonGroup
            # toggles the old and new selection in the same click
            button.toggled.connect(
                lambda checked: self.on_scaling_changed() if checked else None
            )

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Power"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 2.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.valueChanged.connect(self.on_scaling_changed)
        gamma_row.addWidget(self.gamma_spin)
        scaling_layout.addLayout(gamma_row)

        scaling_box.setLayout(scaling_layout)
        return scaling_box

    def get_scaling_fn(self):
        gamma = self.gamma_spin.value()
        if self.linear_radio.isChecked():
            return lambda dp: dp.astype(np.float64, copy=False)
        elif self.log_radio.isChecked():
            return lambda dp: np.log(np.maximum(dp, 1e-6))
        else:
            return lambda dp: np.power(np.maximum(dp, 0), gamma)

    def _append_pane(self):
        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            return

        pane = BraggPreviewPane(f"Preview {len(self.panes) + 1}")
        # sigRegionChanged (not sigRegionChangeFinished) fires continuously
        # while dragging, so the preview updates live as the point selector
        # is moved rather than only once it's released.
        pane.rs_selector.sigRegionChanged.connect(
            partial(self.update_previews, panes=[pane])
        )
        self.panes.append(pane)
        self.previews_layout.addWidget(pane)

        if self.window.probe is not None:
            parent = self.window.parent
            pane.set_realspace_image(parent.get_virtual_image())
            pane.set_marker_diameter(2 * self.window.probe_radius)
            self.update_previews(panes=[pane])

        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            self.add_pane_button.setEnabled(False)

    def add_pane(self):
        self._append_pane()

    def on_probe_accepted(self):
        parent = self.window.parent
        vimg = parent.get_virtual_image()
        diameter = 2 * self.window.probe_radius
        for pane in self.panes:
            pane.set_realspace_image(vimg)
            pane.set_marker_diameter(diameter)
        self.update_previews()

    def _find_peaks(self, dp, xc, yc, probe, params):
        parent = self.window.parent
        try:
            peaks = parent.datacube.find_Bragg_disks(
                template=probe.kernel, data=(xc, yc), **params
            )
            return peaks.qx, peaks.qy
        except Exception as exc:
            parent.statusBar().showMessage(f"Peak finding failed: {exc}", 5_000)
            return np.array([]), np.array([])

    def update_previews(self, panes=None, relevel=False):
        # Called when a pane's point selector moves (position + peaks +
        # display all need refreshing for that pane).
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            return

        panes = panes if panes is not None else self.panes
        params = self.settings_pane.get_params()
        scale_fn = self.get_scaling_fn()

        for pane in panes:
            xc, yc = pane.get_scan_position(parent.datacube)
            dp = parent.datacube.data[xc, yc, :, :]
            qx, qy = self._find_peaks(dp, xc, yc, probe, params)
            pane.update_dp(dp, scale_fn, relevel=relevel)
            pane.update_scatter(qx, qy)

    def on_detection_params_changed(self, *_):
        # Detection-only parameters (thresholds, corrPower, etc.) don't
        # change the diffraction pattern or its display scaling, only which
        # peaks are found -- so just rerun peak-finding against each pane's
        # cached DP, without touching the displayed image or its levels.
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            return

        params = self.settings_pane.get_params()
        for pane in self.panes:
            if pane.last_dp is None:
                continue
            xc, yc = pane.get_scan_position(parent.datacube)
            qx, qy = self._find_peaks(pane.last_dp, xc, yc, probe, params)
            pane.update_scatter(qx, qy)

    def on_scaling_changed(self, *_):
        # Scaling mode/gamma changed -- redisplay each pane's cached DP with
        # freshly computed percentile levels for the new scaling function.
        scale_fn = self.get_scaling_fn()
        for pane in self.panes:
            if pane.last_dp is not None:
                pane.update_dp(pane.last_dp, scale_fn, relevel=True)

    def find_all(self):
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            parent.statusBar().showMessage(
                "Generate/accept a probe kernel first!", 5_000
            )
            return

        params = self.settings_pane.get_params()

        parent.statusBar().showMessage(
            "Running disk detection on the full dataset... "
            "(this may take a while; see console for progress)"
        )
        parent.qtapp.processEvents()

        try:
            parent.datacube.find_Bragg_disks(template=probe.kernel, data=None, **params)
        except Exception as exc:
            parent.statusBar().showMessage(f"Disk detection failed: {exc}", 5_000)
            raise

        parent.statusBar().showMessage(
            "Disk detection complete. Results are attached to the dataset "
            "(export via File > Export Datacube > py4DSTEM HDF5).",
            10_000,
        )
