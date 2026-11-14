"""Every view in one detector image, and the boxes found there back in each view."""
from collections import namedtuple

import torch
import torch.nn.functional as F

PAD = 114 / 255          # the detector's letterbox grey
STRIDE = 32

Layout = namedtuple("Layout", "canvas cell tile scale origins view")
"""Where the views sit: ``canvas``, ``cell`` and ``tile`` are (height, width) in
detector pixels, ``origins`` the (x, y) corner of each view's tile, ``scale``
detector pixels per view pixel along (x, y), ``view`` the (height, width) of a view."""

Detection = namedtuple("Detection", "boxes")


class Boxes:
    """The boxes of one view, with the fields the estimator reads from ultralytics."""

    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = xyxy, conf

    def __len__(self):
        return len(self.conf)


def grid(n_views):
    """(rows, columns): one row up to two views, two columns beyond."""
    if n_views <= 2:
        return 1, n_views
    return -(-n_views // 2), 2


def layout(n_views, view_hw, canvas_hw):
    """Place ``n_views`` images of ``view_hw`` in a detector image of ``canvas_hw``."""
    if canvas_hw[0] % STRIDE or canvas_hw[1] % STRIDE:
        raise ValueError(f"detector image {canvas_hw} is not a multiple of {STRIDE}")
    rows, cols = grid(n_views)
    cell = canvas_hw[0] // rows, canvas_hw[1] // cols
    scale = min(cell[0] / view_hw[0], cell[1] / view_hw[1])
    tile_hw = round(view_hw[0] * scale), round(view_hw[1] * scale)
    origins = [(v % cols * cell[1], v // cols * cell[0]) for v in range(n_views)]
    scales = tile_hw[1] / view_hw[1], tile_hw[0] / view_hw[0]
    return Layout(tuple(canvas_hw), cell, tile_hw, scales, origins, tuple(view_hw))


def tile(images, layout, canvas=None):
    """(C, 3, H, W) images as one (1, 3, *canvas) detector image; ``canvas`` is reused if given."""
    if canvas is None:
        canvas = torch.full((1, 3, *layout.canvas), PAD, dtype=images.dtype, device=images.device)
    small = F.interpolate(images, size=layout.tile, mode="area")
    h, w = layout.tile
    for view, (x0, y0) in enumerate(layout.origins):
        canvas[0, :, y0:y0 + h, x0:x0 + w] = small[view]
    return canvas


def split_boxes(xyxy, conf, layout):
    """Detector-image boxes as one :class:`Detection` per view, in that view's pixels.

    A box belongs to the view whose cell holds its centre and is clipped to it.
    """
    cols = layout.canvas[1] // layout.cell[1]
    centre = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    col = torch.div(centre[:, 0], layout.cell[1], rounding_mode="floor").long()
    row = torch.div(centre[:, 1], layout.cell[0], rounding_mode="floor").long()
    owner = row * cols + col
    origin = torch.stack([col * layout.cell[1], row * layout.cell[0]], dim=1).to(xyxy.dtype)
    local = (xyxy - origin.repeat(1, 2)) / xyxy.new_tensor(layout.scale).repeat(2)
    h, w = layout.view
    local = torch.minimum(local.clamp(min=0), local.new_tensor([w, h, w, h]))
    keep = (local[:, 2] > local[:, 0]) & (local[:, 3] > local[:, 1])
    return [Detection(Boxes(local[keep & (owner == v)], conf[keep & (owner == v)]))
            for v in range(len(layout.origins))]
