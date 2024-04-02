import pyqtgraph as pg
import numpy as np
import py4DSTEM
from functools import partial

from py4D_browser.utils import pg_point_roi, make_detector

def update_real_space_view(self, reset=False):
    scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

    detector_shape = self.detector_shape_group.checkedAction().text().replace("&", "")
    assert detector_shape in [
        "Point",
        "Rectangular",
        "Circle",
        "Annulus",
    ], detector_shape

    detector_mode = self.detector_mode_group.checkedAction().text().replace("&", "")
    assert detector_mode in [
        "Integrating",
        "Maximum",
        "CoM Magnitude",
        "CoM Angle",
        "iCoM",
    ], detector_mode

    # If a CoM method is checked, ensure linear scaling
    if detector_mode in ["CoM Magnitude", "CoM Angle"] and scaling_mode != "Linear":
        print("Warning! Setting linear scaling for CoM image")
        self.vimg_scale_linear_action.setChecked(True)
        scaling_mode = "Linear"

    if self.datacube is None:
        return

    # We will branch through certain combinations of detector shape and mode.
    # If we happen across a special case that can be handled directly, we
    # compute vimg. If we encounter a case that needs a more complicated
    # computation we compute the mask and then do the virtual image later
    mask = None
    if detector_shape == "Rectangular":
        # Get slices corresponding to ROI
        slices, transforms = self.virtual_detector_roi.getArraySlice(
            self.datacube.data[0, 0, :, :], self.diffraction_space_widget.getImageItem()
        )
        slice_y, slice_x = slices

        # update the label:
        self.diffraction_space_view_text.setText(
            f"Diffraction Space Range: [{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
        )

        if detector_mode == "Integrating":
            vimg = np.sum(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
        elif detector_mode == "Maximum":
            vimg = np.max(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
        else:
            mask = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny), dtype=np.bool_)
            mask[slice_x, slice_y] = True

    elif detector_shape == "Circle":
        R = self.virtual_detector_roi.size()[0] / 2.0

        x0 = self.virtual_detector_roi.pos()[0] + R
        y0 = self.virtual_detector_roi.pos()[1] + R

        self.diffraction_space_view_text.setText(
            f"Detector Center: ({x0:.0f},{y0:.0f}), Radius: {R:.0f}"
        )

        mask = make_detector(
            (self.datacube.Q_Nx, self.datacube.Q_Ny), "circle", ((x0, y0), R)
        )
    elif detector_shape == "Annulus":
        inner_pos = self.virtual_detector_roi_inner.pos()
        inner_size = self.virtual_detector_roi_inner.size()
        R_inner = inner_size[0] / 2.0
        x0 = inner_pos[0] + R_inner
        y0 = inner_pos[1] + R_inner

        outer_size = self.virtual_detector_roi_outer.size()
        R_outer = outer_size[0] / 2.0

        if R_inner <= R_outer:
            R_inner -= 1

        self.diffraction_space_view_text.setText(
            f"Detector Center: ({x0:.0f},{y0:.0f}), Radii: ({R_inner:.0f},{R_outer:.0f})"
        )

        mask = make_detector(
            (self.datacube.Q_Nx, self.datacube.Q_Ny),
            "annulus",
            ((x0, y0), (R_inner, R_outer)),
        )
    elif detector_shape == "Point":
        roi_state = self.virtual_detector_point.saveState()
        y0, x0 = roi_state["pos"]
        xc, yc = int(x0 + 1), int(y0 + 1)

        # Set the diffraction space image
        # Normalize coordinates
        xc = np.clip(xc, 0, self.datacube.Q_Nx - 1)
        yc = np.clip(yc, 0, self.datacube.Q_Ny - 1)
        vimg = self.datacube.data[:, :, xc, yc]

        self.diffraction_space_view_text.setText(f"Diffraction Pixel: [{xc},{yc}]")

    else:
        raise ValueError("Detector shape not recognized")

    if mask is not None:
        # For debugging masks:
        # self.diffraction_space_widget.setImage(
        #     mask.T, autoLevels=True, autoRange=True
        # )
        mask = mask.astype(np.float32)
        vimg = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny))
        iterator = py4DSTEM.tqdmnd(self.datacube.R_Nx, self.datacube.R_Ny, disable=True)

        if detector_mode == "Integrating":
            for rx, ry in iterator:
                vimg[rx, ry] = np.sum(self.datacube.data[rx, ry] * mask)

        elif detector_mode == "Maximum":
            for rx, ry in iterator:
                vimg[rx, ry] = np.max(self.datacube.data[rx, ry] * mask)

        elif "CoM" in detector_mode:
            ry_coord, rx_coord = np.meshgrid(
                np.arange(self.datacube.Q_Ny), np.arange(self.datacube.Q_Nx)
            )
            CoMx = np.zeros_like(vimg)
            CoMy = np.zeros_like(vimg)
            for rx, ry in iterator:
                ar = self.datacube.data[rx, ry] * mask
                tot_intens = np.sum(ar)
                CoMx[rx, ry] = np.sum(rx_coord * ar) / tot_intens
                CoMy[rx, ry] = np.sum(ry_coord * ar) / tot_intens

            CoMx -= np.mean(CoMx)
            CoMy -= np.mean(CoMy)

            if detector_mode == "CoM Magnitude":
                vimg = np.hypot(CoMx, CoMy)
            elif detector_mode == "CoM Angle":
                vimg = np.arctan2(CoMy, CoMx)
            elif detector_mode == "iCoM":
                dpc = py4DSTEM.process.phase.DPC(verbose=False)
                dpc.preprocess(
                    force_com_measured=[CoMx, CoMy],
                    plot_rotation=False,
                    plot_center_of_mass="",
                )
                dpc.reconstruct(max_iter=1, step_size=1)
                vimg = dpc.object_phase
            else:
                raise ValueError("Mode logic gone haywire!")

        else:
            raise ValueError("Oopsie")

    if scaling_mode == "Linear":
        new_view = vimg
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(vimg, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(vimg, 0))
    else:
        raise ValueError("Mode not recognized")

    self.unscaled_realspace_image = vimg

    self.real_space_widget.setImage(
        new_view.T,
        autoLevels=reset or self.realspace_rescale_button.latched,
        autoRange=reset,
    )

    # Update FFT view
    if self.fft_source_action_group.checkedAction().text() == "Virtual Image FFT":
        fft = np.abs(np.fft.fftshift(np.fft.fft2(new_view))) ** 0.5
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.fft_widget_text.setText("Virtual Image FFT")
        self.fft_widget.setImage(
            fft.T, autoLevels=False, levels=levels, autoRange=mode_switch
        )
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            # Need to autorange after setRect
            self.fft_widget.autoRange()


def update_diffraction_space_view(self, reset=False):
    scaling_mode = self.diff_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"]

    if self.datacube is None:
        return

    detector_shape = (
        self.rs_detector_shape_group.checkedAction().text().replace("&", "")
    )
    assert detector_shape in [
        "Point",
        "Rectangular",
    ], detector_shape

    if detector_shape == "Point":
        roi_state = self.real_space_point_selector.saveState()
        y0, x0 = roi_state["pos"]
        xc, yc = int(x0 + 1), int(y0 + 1)

        # Set the diffraction space image
        # Normalize coordinates
        xc = np.clip(xc, 0, self.datacube.R_Nx - 1)
        yc = np.clip(yc, 0, self.datacube.R_Ny - 1)

        self.real_space_view_text.setText(f"Real Space Pixel: [{xc},{yc}]")

        DP = self.datacube.data[xc, yc]
    elif detector_shape == "Rectangular":
        # Get slices corresponding to ROI
        slices, _ = self.real_space_rect_selector.getArraySlice(
            np.zeros((self.datacube.Rshape)), self.real_space_widget.getImageItem()
        )
        slice_y, slice_x = slices

        # update the label:
        self.real_space_view_text.setText(
            f"Real Space Range: [{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
        )

        DP = np.sum(self.datacube.data[slice_x, slice_y], axis=(0, 1))

    else:
        raise ValueError("Detector shape not recognized")

    self.unscaled_diffraction_image = DP

    if scaling_mode == "Linear":
        new_view = DP
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(DP, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(DP, 0))
    else:
        raise ValueError("Mode not recognized")

    self.diffraction_space_widget.setImage(
        new_view.T,
        autoLevels=reset or self.diffraction_rescale_button.latched,
        autoRange=reset,
    )

    if self.fft_source_action_group.checkedAction().text() == "EWPC":
        log_clip = np.maximum(1e-10, np.percentile(np.maximum(DP, 0.0), 0.1))
        fft = np.abs(np.fft.fftshift(np.fft.fft2(np.log(np.maximum(DP, log_clip)))))
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "EWPC"
        self.fft_widget_text.setText("EWPC")
        self.fft_widget.setImage(
            fft.T, autoLevels=False, levels=levels, autoRange=mode_switch
        )

    # remove ROIs if button is unlatched to return to the default view
    if self.disk_group in self.diffraction_space_widget.view.allChildren():
        self.diffraction_space_widget.removeItem(self.disk_group)
    
    # Continuously update disk positions as real space ROI moves when button
    # is latched
    if self.separate_window is not None:
        if self.separate_window.run_all.isChecked():
            self.update_disk_detection()

def update_realspace_detector(self):
    # change the shape of the detector, then update the view

    detector_shape = (
        self.rs_detector_shape_group.checkedAction().text().replace("&", "")
    )
    assert detector_shape in ["Point", "Rectangular"], detector_shape

    if self.datacube is None:
        return

    x, y = self.datacube.shape[2:]
    x0, y0 = x / 2, y / 2
    xr, yr = x / 10, y / 10

    # Remove existing detector
    if hasattr(self, "real_space_point_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_point_selector)
        self.real_space_point_selector = None
    if hasattr(self, "real_space_rect_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_rect_selector)
        self.real_space_rect_selector = None

    # Rectangular detector
    if detector_shape == "Point":
        self.real_space_point_selector = pg_point_roi(self.real_space_widget.getView())
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )

    elif detector_shape == "Rectangular":
        self.real_space_rect_selector = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.real_space_widget.getView().addItem(self.real_space_rect_selector)
        self.real_space_rect_selector.sigRegionChangeFinished.connect(
            partial(self.update_diffraction_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_diffraction_space_view(reset=True)


def update_diffraction_detector(self):
    # change the shape of the detector, then update the view

    detector_shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert detector_shape in ["Point", "Rectangular", "Circle", "Annulus"]

    if self.datacube is None:
        return

    x, y = self.datacube.shape[2:]
    x0, y0 = x / 2, y / 2
    xr, yr = x / 10, y / 10

    # Remove existing detector
    if hasattr(self, "virtual_detector_point"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_point
        )
        self.virtual_detector_point = None
    if hasattr(self, "virtual_detector_roi"):
        self.diffraction_space_widget.view.scene().removeItem(self.virtual_detector_roi)
        self.virtual_detector_roi = None
    if hasattr(self, "virtual_detector_roi_inner"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_inner
        )
        self.virtual_detector_roi_inner = None
    if hasattr(self, "virtual_detector_roi_outer"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_outer
        )
        self.virtual_detector_roi_outer = None

    # Rectangular detector
    if detector_shape == "Point":
        self.virtual_detector_point = pg_point_roi(
            self.diffraction_space_widget.getView()
        )
        self.virtual_detector_point.sigRegionChanged.connect(
            partial(self.update_real_space_view, False)
        )

    elif detector_shape == "Rectangular":
        self.virtual_detector_roi = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Circular detector
    elif detector_shape == "Circle":
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Annular dector
    elif detector_shape == "Annulus":
        # Make outer detector
        self.virtual_detector_roi_outer = pg.CircleROI(
            [int(x0 - xr), int(y0 - yr)], [int(2 * xr), int(2 * yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_outer)

        # Make inner detector
        self.virtual_detector_roi_inner = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=(4, 9),
            movable=False,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_inner)

        # Connect size/position of inner and outer detectors
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_pos
        )
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_radii
        )
        self.virtual_detector_roi_inner.sigRegionChanged.connect(
            self.update_annulus_radii
        )

        # Connect to real space view update function
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_real_space_view(reset=True)


def nudge_real_space_selector(self, dx, dy):
    if (
        hasattr(self, "real_space_point_selector")
        and self.real_space_point_selector is not None
    ):
        selector = self.real_space_point_selector
    elif (
        hasattr(self, "real_space_rect_selector")
        and self.real_space_rect_selector is not None
    ):
        selector = self.real_space_rect_selector
    else:
        raise RuntimeError("Can't find the real space selector!")

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def nudge_diffraction_selector(self, dx, dy):
    if (
        hasattr(self, "virtual_detector_point")
        and self.virtual_detector_point is not None
    ):
        selector = self.virtual_detector_point
    elif (
        hasattr(self, "virtual_detector_roi") and self.virtual_detector_roi is not None
    ):
        selector = self.virtual_detector_roi
    elif (
        hasattr(self, "virtual_detector_roi_outer")
        and self.virtual_detector_roi_outer is not None
    ):
        selector = self.virtual_detector_roi_outer
    else:
        raise RuntimeError("Can't find the diffraction space selector!")

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def update_annulus_pos(self):
    """
    Function to keep inner and outer rings of annulus aligned.
    """
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    # Only outer annulus is draggable; when it moves, update position of inner annulus
    x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
    y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
    self.virtual_detector_roi_inner.setPos(x0 - R_inner, y0 - R_inner, update=False)


def update_annulus_radii(self):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    if R_outer < R_inner:
        x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
        y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
        self.virtual_detector_roi_outer.setSize(2 * R_inner + 6, update=False)
        self.virtual_detector_roi_outer.setPos(
            x0 - R_inner - 3, y0 - R_inner - 3, update=False
        )

def update_probe_template_view(self, _=None):
        
    if hasattr(self, "real_space_point_selector"):
        roi = self.real_space_point_selector
    if hasattr(self, "real_space_rect_selector"):
        roi = self.real_space_rect_selector

    pos = roi.pos()
    size = roi.size()

    # Create a mask that is the same shape as the data and is 0 everywhere
    mask = np.zeros(self.datacube.Rshape, dtype=bool)

    # Set the region of the mask under the ROI to 1
    mask[int(pos[1]):int(pos[1]+size[1]), int(pos[0]):int(pos[0]+size[0])] = 1
    print(np.count_nonzero(mask), "pixels in ROI")
    # Use current ROI in real space to generate a probe template
    self.probe = self.datacube.get_vacuum_probe(mask)
    
    self.probe_view.setImage(self.probe.probe)
    
    self.alpha_pr, self.qx0_pr, self.qy0_pr = self.datacube.get_probe_size(self.probe.probe)
    print(f"Probe size: {self.alpha_pr} px"
          f" at ({self.qx0_pr}, {self.qy0_pr}) px")

def update_kernel_view(self):
    # Update the kernel view
    multiplier = self.kernel_radius.value()
    self.probe.get_kernel(
        mode='sigmoid',
        origin=(self.qx0_pr, self.qy0_pr),
        radii=(self.alpha_pr, multiplier*self.alpha_pr)   # the inner and outer radii of the 'trench'
    )
    
    R = 24
    kernel = self.probe.kernel
    im_kernel = np.vstack(
        [
            np.hstack([kernel[-int(R) :, -int(R) :], kernel[-int(R) :, : int(R)]]),
            np.hstack([kernel[: int(R), -int(R) :], kernel[: int(R), : int(R)]]),
        ]
    )
    
    self.kernel_view.setImage(im_kernel)

def update_disk_detection(self):
    """
    Finds Bragg disks for the currently displayed diffraction pattern.
    """
    
    # Remove existing CircleROIs
    for item in self.disk_group.childItems():
        self.disk_group.removeFromGroup(item)

    # take current real space ROI position and turn it into pixel coordinates
    # this part was taken from one of the functions above
    roi_state = self.real_space_point_selector.saveState()
    y0, x0 = roi_state["pos"]
    xc, yc = int(x0 + 1), int(y0 + 1)

    # Normalize coordinates
    xc = np.clip(xc, 0, self.datacube.R_Nx - 1)
    yc = np.clip(yc, 0, self.datacube.R_Ny - 1)
    
    # get parameters in dictionary form from the parameter window
    detection_params = self.separate_window.get_params_as_dict()

    braggpeaks = self.datacube.find_Bragg_disks(
        data=(xc, yc),
        template=self.probe.kernel,
        **detection_params,
    )
    
    for qx, qy in zip(braggpeaks.qx, braggpeaks.qy):
        
        disk_roi = pg.CircleROI(
            (qy-self.alpha_pr/2+0.5, qx-self.alpha_pr/2+0.5),
            (self.alpha_pr, self.alpha_pr),
            movable=False,
            resizable=False,
            pen=pg.mkPen('r', width=2, cosmetic=True)
        )
        self.disk_group.addToGroup(disk_roi)
    
    self.diffraction_space_widget.addItem(self.disk_group)
 