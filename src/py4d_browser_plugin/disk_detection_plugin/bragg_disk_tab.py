from functools import partial

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.utils import pg_point_roi

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer
    from .disk_detection_plugin import DiskDetectionWindow


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
            # Transposed to match the main window's real_space_widget
            # convention (see _render_virtual_image in update_views.py) --
            # otherwise non-square scan shapes appear flipped/rotated
            # relative to the main virtual image view, and get_scan_position
            # below (which mirrors the main window's own point-selector
            # reading convention) would read back the wrong (xc, yc).
            self.rs_view.setImage(image.T, autoLevels=True, autoRange=True)

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
        # Transposed to match the main window's diffraction_space_widget
        # convention (see _render_diffraction_image in update_views.py).
        scaled = scale_fn(dp).T

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
        # dp is displayed transposed (see update_dp), so marker positions
        # need the same (y, x) swap to stay aligned with the disks --
        # verified empirically with an offscreen render against a synthetic
        # test image.
        spots = [{"pos": [y, x], "data": 1} for x, y in zip(qx, qy)]
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

        # Off by default: previews only update when a point selector drag is
        # released. When checked, previews also update continuously while
        # dragging (can be too slow/jumpy to want on all the time).
        self.live_update_checkbox = QCheckBox("Live Update While Dragging")
        self.live_update_checkbox.setChecked(False)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.settings_pane)
        left_layout.addWidget(scaling_box)
        left_layout.addWidget(self.find_all_button)
        left_layout.addWidget(self.add_pane_button)
        left_layout.addWidget(self.live_update_checkbox)
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
        # Always update once the drag is released. Also update continuously
        # during the drag (sigRegionChanged), but only if the user has
        # opted into that via the "Live Update While Dragging" checkbox.
        pane.rs_selector.sigRegionChangeFinished.connect(
            partial(self.update_previews, panes=[pane])
        )
        pane.rs_selector.sigRegionChanged.connect(
            partial(self._on_pane_region_changed, pane)
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

    def _on_pane_region_changed(self, pane, *_):
        if self.live_update_checkbox.isChecked():
            self.update_previews(panes=[pane])

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
