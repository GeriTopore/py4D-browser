import os

import h5py
import numpy as np
import pyqtgraph as pg
import py4DSTEM
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
        source_box = QGroupBox("Vacuum Region")
        source_layout = QVBoxLayout()

        button_row = QHBoxLayout()
        use_selection_button = QPushButton("Use Current Selection")
        use_selection_button.clicked.connect(self.use_current_selection)
        button_row.addWidget(use_selection_button)

        load_vacuum_button = QPushButton("Load Vacuum File...")
        load_vacuum_button.clicked.connect(self.load_vacuum_file)
        button_row.addWidget(load_vacuum_button)
        source_layout.addLayout(button_row)

        self.source_label = QLabel("No vacuum region selected yet")
        source_layout.addWidget(self.source_label)
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
        self._source = "selection"
        self._vacuum_datacube = None
        self.source_label.setText("Source: rectangular selection on current dataset")

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
        self._source = "vacuum_file"
        self.source_label.setText(f"Source: {os.path.basename(filepath)}")

    def generate_probe(self):
        parent = self.window.parent
        if parent.datacube is None:
            parent.statusBar().showMessage(
                "Load a dataset in the main window first!", 5_000
            )
            return

        threshold = self.threshold_spin.value()
        expansion = self.expansion_spin.value()
        opening = self.opening_spin.value()

        if self._source == "selection":
            detector: DetectorInfo = parent.get_virtual_image_detector()
            if detector["shape"] is not DetectorShape.RECTANGULAR:
                parent.statusBar().showMessage(
                    "Select a rectangular region on the virtual image to use as "
                    "the vacuum region.",
                    5_000,
                )
                return
            probe_source = parent.datacube
            roi_kwargs = {"ROI": detector["mask"]}
        elif self._source == "vacuum_file":
            probe_source = self._vacuum_datacube
            roi_kwargs = {}
        else:
            parent.statusBar().showMessage(
                "Choose a vacuum region source first!", 5_000
            )
            return

        self.probe = probe_source.get_vacuum_probe(
            threshold=threshold, expansion=expansion, opening=opening, **roi_kwargs
        )
        self.alpha, self.qx0, self.qy0 = parent.datacube.get_probe_size(
            self.probe.probe
        )

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
        self.sigma_cc_spin.setValue(2.0)
        form.addRow("Correlation Smoothing Sigma", self.sigma_cc_spin)

        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["pixel", "poly", "multicorr"])
        self.subpixel_combo.setCurrentText("multicorr")
        form.addRow("Subpixel Mode", self.subpixel_combo)

        self.upsample_factor_spin = QSpinBox()
        self.upsample_factor_spin.setRange(1, 256)
        self.upsample_factor_spin.setValue(16)
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
        self.min_rel_intensity_spin.setValue(0.005)
        form.addRow("Minimum Relative Intensity", self.min_rel_intensity_spin)

        self.relative_to_peak_spin = QSpinBox()
        self.relative_to_peak_spin.setRange(0, 20)
        self.relative_to_peak_spin.setValue(0)
        form.addRow("Relative to Peak #", self.relative_to_peak_spin)

        self.min_peak_spacing_spin = QSpinBox()
        self.min_peak_spacing_spin.setRange(0, 1000)
        self.min_peak_spacing_spin.setValue(60)
        form.addRow("Minimum Peak Spacing (px)", self.min_peak_spacing_spin)

        self.edge_boundary_spin = QSpinBox()
        self.edge_boundary_spin.setRange(0, 1000)
        self.edge_boundary_spin.setValue(20)
        form.addRow("Edge Boundary (px)", self.edge_boundary_spin)

        self.max_num_peaks_spin = QSpinBox()
        self.max_num_peaks_spin.setRange(1, 1000)
        self.max_num_peaks_spin.setValue(70)
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

    def update_dp_and_scatter(self, dp, qx, qy):
        # Square-root scaling (matching the main window's default diffraction
        # scaling) plus percentile-based levels, so the faint diffracted
        # disks remain visible alongside the much brighter central beam.
        scaled = np.sqrt(np.maximum(dp, 0))
        levels = tuple(np.percentile(scaled, [0.1, 99.9]))
        self.dp_view.setImage(scaled, autoLevels=False, levels=levels, autoRange=True)
        # ScatterPlotItem positions map directly onto the array indices
        # ImageView.setImage was given -- no axis swap needed here (verified
        # empirically; the old interactive_disk_detection branch's swap was
        # a leftover from a different image-display setup and is wrong here).
        spots = [{"pos": [x, y], "data": 1} for x, y in zip(qx, qy)]
        self.scatter.setData(spots)


class BraggDiskTab(QWidget):
    def __init__(self, window: "DiskDetectionWindow"):
        super().__init__()

        self.window = window

        self.settings_pane = BraggDiskSettingsPane()
        self.settings_pane.connect_changed(self.update_previews)

        self.find_all_button = QPushButton("Find All Bragg Disks")
        self.find_all_button.clicked.connect(self.find_all)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.settings_pane)
        left_layout.addWidget(self.find_all_button)
        left_layout.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self.panes = [BraggPreviewPane(f"Preview {i + 1}") for i in range(3)]
        for pane in self.panes:
            pane.rs_selector.sigRegionChangeFinished.connect(self.update_previews)

        previews_layout = QHBoxLayout()
        for pane in self.panes:
            previews_layout.addWidget(pane)
        previews_widget = QWidget()
        previews_widget.setLayout(previews_layout)

        layout = QHBoxLayout()
        layout.addWidget(left_widget, 1)
        layout.addWidget(previews_widget, 4)
        self.setLayout(layout)

    def on_probe_accepted(self):
        parent = self.window.parent
        vimg = parent.get_virtual_image()
        diameter = 2 * self.window.probe_radius
        for pane in self.panes:
            pane.set_realspace_image(vimg)
            pane.set_marker_diameter(diameter)
        self.update_previews()

    def update_previews(self, *_):
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            return

        params = self.settings_pane.get_params()

        for pane in self.panes:
            xc, yc = pane.get_scan_position(parent.datacube)
            dp = parent.datacube.data[xc, yc, :, :]

            try:
                peaks = parent.datacube.find_Bragg_disks(
                    template=probe.kernel, data=(xc, yc), **params
                )
                qx, qy = peaks.qx, peaks.qy
            except Exception as exc:
                parent.statusBar().showMessage(f"Peak finding failed: {exc}", 5_000)
                qx, qy = np.array([]), np.array([])

            pane.update_dp_and_scatter(dp, qx, qy)

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
