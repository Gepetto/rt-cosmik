"""Tiled detection: the views in one detector image, and its boxes back in each view."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.nlf import tiling                                      # noqa: E402

VIEW = (720, 1280)


def to_canvas(box, view, layout):
    """A view-pixel xyxy box in detector-image pixels."""
    x0, y0 = layout.origins[view]
    sx, sy = layout.tile[1] / layout.view[1], layout.tile[0] / layout.view[0]
    return [box[0] * sx + x0, box[1] * sy + y0, box[2] * sx + x0, box[3] * sy + y0]


def split(boxes, layout, conf=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    conf = torch.full((len(boxes),), 0.9) if conf is None else torch.tensor(conf)
    return tiling.split_boxes(boxes, conf, layout)


@pytest.mark.parametrize("views, grid", [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2))])
def test_two_views_share_a_row_and_three_or_four_a_2x2_grid(views, grid):
    assert tiling.grid(views) == grid


def test_four_views_each_take_a_quarter_of_a_square_detector_image():
    layout = tiling.layout(4, VIEW, (640, 640))
    assert layout.scale == pytest.approx((0.25, 0.25))
    assert layout.tile == (180, 320)
    assert layout.origins == [(0, 0), (320, 0), (0, 320), (320, 320)]


def test_a_detector_image_shaped_like_the_grid_keeps_more_of_each_view():
    layout = tiling.layout(4, VIEW, (736, 1280))
    assert layout.scale == pytest.approx((0.5, 0.5))
    assert layout.origins == [(0, 0), (640, 0), (0, 368), (640, 368)]


def test_two_views_sit_side_by_side():
    layout = tiling.layout(2, VIEW, (640, 640))
    assert layout.origins == [(0, 0), (320, 0)]
    assert layout.scale == pytest.approx((0.25, 0.25))


def test_each_view_lands_in_its_own_tile_and_the_rest_is_padding():
    images = torch.stack([torch.full((3, *VIEW), v) for v in (0.2, 0.4, 0.6)])
    layout = tiling.layout(3, VIEW, (640, 640))
    canvas = tiling.tile(images, layout)
    assert canvas.shape == (1, 3, 640, 640)
    for value, (x0, y0) in zip((0.2, 0.4, 0.6), layout.origins):
        assert torch.allclose(canvas[0, :, y0:y0 + 180, x0:x0 + 320], torch.tensor(value))
    assert torch.allclose(canvas[0, :, 180:320], torch.tensor(tiling.PAD))
    assert torch.allclose(canvas[0, :, 320:, 320:], torch.tensor(tiling.PAD))


def test_tiling_reuses_the_canvas_it_is_given():
    images = torch.rand(2, 3, *VIEW)
    layout = tiling.layout(2, VIEW, (640, 640))
    canvas = tiling.tile(images, layout)
    assert tiling.tile(images, layout, canvas) is canvas


def test_tile_keeps_the_image_content_in_place():
    images = torch.zeros(4, 3, *VIEW)
    images[1, :, :360, 640:] = 1.0                        # top-right quarter of view 1
    canvas = tiling.tile(images, tiling.layout(4, VIEW, (640, 640)))
    assert canvas[0, :, 0:90, 480:640].min() > 0.99
    assert canvas[0, :, 90:180, 320:480].max() < 0.01


@pytest.mark.parametrize("canvas", [(640, 640), (736, 1280)])
def test_a_box_returns_to_the_pixels_of_its_view(canvas):
    layout = tiling.layout(4, VIEW, canvas)
    boxes = [[100.0, 50.0, 400.0, 700.0], [900.0, 10.0, 1200.0, 600.0],
             [0.0, 0.0, 1280.0, 720.0], [640.0, 200.0, 800.0, 640.0]]
    found = split([to_canvas(b, v, layout) for v, b in enumerate(boxes)], layout)
    assert len(found) == 4
    for view, box in enumerate(boxes):
        assert len(found[view].boxes) == 1
        assert torch.allclose(found[view].boxes.xyxy[0], torch.tensor(box), atol=1e-2)


def test_a_box_returns_exactly_when_the_tile_is_rounded_to_whole_pixels():
    layout = tiling.layout(4, (480, 854), (640, 640))     # 179.86 rows of tile, rounded to 180
    assert layout.tile == (180, 320)
    box = [100.0, 50.0, 400.0, 470.0]
    found = split([to_canvas(box, 2, layout)], layout)
    assert torch.allclose(found[2].boxes.xyxy[0], torch.tensor(box), atol=1e-2)


def test_a_box_goes_to_the_view_holding_its_centre_and_is_clipped_to_it():
    layout = tiling.layout(4, VIEW, (640, 640))
    found = split([[300.0, 20.0, 360.0, 100.0],           # centre x 330: view 1, starts in view 0
                   [200.0, 100.0, 300.0, 400.0]], layout)  # centre y 250: view 0, runs past it
    assert torch.allclose(found[1].boxes.xyxy[0], torch.tensor([0.0, 80.0, 160.0, 400.0]))
    assert torch.allclose(found[0].boxes.xyxy[0], torch.tensor([800.0, 400.0, 1200.0, 720.0]))
    assert len(found[2].boxes) == 0 and len(found[3].boxes) == 0


def test_boxes_outside_every_view_are_dropped():
    layout = tiling.layout(3, VIEW, (640, 640))
    found = split([[400.0, 400.0, 500.0, 480.0],          # the empty fourth cell
                   [10.0, 200.0, 100.0, 300.0]], layout)   # the padding under view 0
    assert [len(d.boxes) for d in found] == [0, 0, 0]


def test_every_person_of_a_view_is_returned_with_its_confidence():
    layout = tiling.layout(2, VIEW, (640, 640))
    found = split([[10.0, 10.0, 60.0, 150.0], [330.0, 10.0, 400.0, 170.0],
                   [200.0, 20.0, 260.0, 160.0]], layout, conf=[0.5, 0.8, 0.7])
    assert found[0].boxes.conf.tolist() == pytest.approx([0.5, 0.7])
    assert found[1].boxes.conf.tolist() == pytest.approx([0.8])
    assert found[0].boxes.xyxy.shape == (2, 4)


def test_no_detection_gives_an_empty_result_per_view():
    found = split([], tiling.layout(4, VIEW, (640, 640)))
    assert [d.boxes.xyxy.shape for d in found] == [(0, 4)] * 4


def test_detector_image_must_be_a_multiple_of_the_stride():
    with pytest.raises(ValueError, match="32"):
        tiling.layout(4, VIEW, (720, 1280))
