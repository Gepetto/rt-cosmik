"""The TensorRT backend's CPU-side pieces: which vertices an engine holds, and the crop geometry."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik import landmark_maps                                   # noqa: E402
from rtcosmik.config_loader import settings                          # noqa: E402
from rtcosmik.nlf import crops                                       # noqa: E402
from rtcosmik.nlf import crop_network                                # noqa: E402
from rtcosmik.nlf.crop_network import inv3x3, landmark_rows          # noqa: E402

K = torch.tensor([[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]])


def test_one_engine_serves_every_landmark_map():
    vertices = landmark_maps.all_vertices(settings)
    maps = [landmark_maps.load("nlf", fold) for fold in range(6)]
    maps.append(landmark_maps.submitted(settings))
    queried = [landmark_maps.nlf_indices(m, settings) for m in maps]
    queried.append(settings.toolbox_nlf_indices)
    for indices in queried:
        rows = landmark_rows(vertices, indices)
        assert [vertices[r] for r in rows] == list(indices)


def test_toolbox_map_is_the_43_vertices_of_settings_py():
    assert len(settings.toolbox_nlf_indices) == 43
    assert settings.toolbox_nlf_indices[4] == 5484        # C7 before the paper's fitted 3353


def test_engine_without_a_queried_vertex_is_refused():
    with pytest.raises(KeyError, match="export_nlf_engine"):
        landmark_rows([10, 20, 30], [20, 40])


def test_crop_of_a_centred_box_zooms_its_longest_side_to_the_crop():
    K_crop, R, zoom = crops.crop_cameras(K, torch.tensor([[540.0, 160.0, 200.0, 400.0]]))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)
    assert zoom.item() == pytest.approx(256 / 400)
    assert K_crop[0, 0, 0].item() == pytest.approx(640.0)
    assert K_crop[0, :2, 2].tolist() == [128.0, 128.0]


def test_crop_camera_looks_at_the_box_centre():
    box = torch.tensor([[900.0, 100.0, 150.0, 300.0]])
    K_crop, R, _ = crops.crop_cameras(K, box)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3), atol=1e-5)
    assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-5)
    centre_ray = torch.linalg.inv(K[0]) @ torch.tensor([975.0, 250.0, 1.0])
    seen = K_crop[0] @ R[0] @ centre_ray
    assert torch.allclose(seen[:2] / seen[2], torch.tensor([128.0, 128.0]), atol=1e-3)


def test_landmark_on_the_crop_axis_returns_to_the_box_centre():
    box = torch.tensor([[900.0, 100.0, 150.0, 300.0]])
    _, R, _ = crops.crop_cameras(K, box)
    poses3d, poses2d = crops.to_camera_frame(torch.tensor([[[0.0, 0.0, 3000.0]]]), R, K)
    assert torch.linalg.norm(poses3d).item() == pytest.approx(3000.0, rel=1e-5)
    assert torch.allclose(poses2d[0, 0], torch.tensor([975.0, 250.0]), atol=1e-2)


def test_crop_shows_the_image_region_of_its_box():
    image = torch.zeros(1, 3, 720, 1280)
    image[:, 0, :, 640:] = 1.0                            # red right of the principal point
    cam = crops.crop_cameras(K, torch.tensor([[540.0, 160.0, 200.0, 400.0]]))
    crop = crops.perspective_crops(image, K, *cam, [0])
    assert crop.shape == (1, 3, 256, 256)
    assert crop[0, 0, :, :120].max() < 0.01 and crop[0, 0, :, 136:].min() > 0.99
    assert crop[0, 1].max() < 0.01


def test_inv3x3_inverts_an_intrinsic_matrix():
    expected = torch.tensor([[[0.001, 0.0, -0.64], [0.0, 0.001, -0.36], [0.0, 0.0, 1.0]]])
    assert torch.allclose(inv3x3(K), expected, atol=1e-7)


META = {"vertices": [10, 20], "precision": "fp16", "max_crops": 8, "tensorrt": "10.0"}


def test_sidecar_names_the_engine_it_was_written_with(tmp_path):
    engine = tmp_path / "nlf.engine"
    crop_network.write_engine(engine, b"engine bytes", META)
    meta = crop_network.checked_meta(engine, engine.read_bytes(), "10.0")
    assert meta["vertices"] == [10, 20] and len(meta["engine_sha256"]) == 64
    assert sorted(f.name for f in tmp_path.iterdir()) == ["nlf.engine", "nlf.engine.json"]


def test_sidecar_of_another_engine_is_refused(tmp_path):
    engine = tmp_path / "nlf.engine"
    crop_network.write_engine(engine, b"engine bytes", META)
    with pytest.raises(RuntimeError, match="not the sidecar of this engine.*export_nlf_engine"):
        crop_network.checked_meta(engine, b"another engine", "10.0")


def test_sidecar_without_the_engine_hash_is_refused(tmp_path):
    engine = tmp_path / "nlf.engine"
    engine.write_bytes(b"engine bytes")
    crop_network.meta_path(engine).write_text(json.dumps(META))
    with pytest.raises(RuntimeError, match="export_nlf_engine"):
        crop_network.checked_meta(engine, b"engine bytes", "10.0")


def test_engine_without_a_sidecar_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="no sidecar.*export_nlf_engine"):
        crop_network.checked_meta(tmp_path / "nlf.engine", b"engine bytes", "10.0")


def test_engine_built_with_another_tensorrt_is_refused(tmp_path):
    engine = tmp_path / "nlf.engine"
    crop_network.write_engine(engine, b"engine bytes", META)
    with pytest.raises(RuntimeError, match="TensorRT 10.0.*10.3.*export_nlf_engine"):
        crop_network.checked_meta(engine, b"engine bytes", "10.3")


def test_more_crops_than_the_engine_takes_are_refused_with_both_numbers():
    engine = object.__new__(crop_network.CropEngine)
    engine.max_crops = 8
    with pytest.raises(ValueError, match="9 crops.*at most 8"):
        engine(torch.zeros(9, 3, 256, 256), torch.eye(3).repeat(9, 1, 1), torch.arange(3))


def test_file_that_is_not_an_engine_is_refused(tmp_path):
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    engine = tmp_path / "nlf.engine"
    crop_network.write_engine(engine, b"not an engine", {**META, "tensorrt": trt.__version__})
    with pytest.raises(RuntimeError, match="could not load.*export_nlf_engine"):
        crop_network.CropEngine(engine)
