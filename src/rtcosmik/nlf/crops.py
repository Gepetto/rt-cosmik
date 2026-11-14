"""NLF's perspective crop: each detected person seen by a virtual camera turned towards them."""
import torch
import torch.nn.functional as F

SIDE = 256
GAMMA = 0.8          # the release's gamma for a single pass
PYRAMID_LEVELS = 3


def crop_cameras(K, boxes, side=SIDE):
    """Virtual camera of each box.

    Args:
        K: (B, 3, 3) intrinsics of the view each box is in.
        boxes: (B, 4) boxes as x, y, width, height, in pixels.

    Returns:
        ``(K_crop, R, zoom)``: (B, 3, 3) crop intrinsics, (B, 3, 3) rotation
        taking camera-frame points to the crop camera, (B,) pixel zoom.
    """
    x, y, w, h = boxes.unbind(1)
    cx, cy = x + w / 2, y + h / 2
    points = torch.stack((torch.stack((cx, cy), 1), torch.stack((cx, y), 1),
                          torch.stack((x + w, cy), 1), torch.stack((cx, y + h), 1),
                          torch.stack((x, cy), 1)), 1)
    rays = F.pad(points, (0, 1), value=1.0) @ torch.linalg.inv(K).transpose(1, 2)
    rays = F.pad(rays[..., :2] / rays[..., 2:], (0, 1), value=1.0)

    up = torch.tensor([0.0, -1.0, 0.0], dtype=K.dtype, device=K.device).expand(len(K), 3)
    new_z = rays[:, 0] / torch.linalg.norm(rays[:, 0], dim=-1, keepdim=True)
    new_x = torch.linalg.cross(new_z, up)
    sideways = torch.stack((new_z[:, 2], torch.zeros_like(new_z[:, 2]), -new_z[:, 0]), 1)
    new_x = torch.where(torch.linalg.norm(new_x, dim=-1, keepdim=True) == 0, sideways, new_x)
    new_x = new_x / torch.linalg.norm(new_x, dim=-1, keepdim=True)
    R = torch.stack((new_x, torch.linalg.cross(new_z, new_x), new_z), 1)

    sides = rays[:, 1:] @ (K @ R).transpose(1, 2)
    sides = sides[..., :2] / sides[..., 2:]
    size = torch.maximum(torch.linalg.norm(sides[:, 0] - sides[:, 2], dim=-1),
                         torch.linalg.norm(sides[:, 1] - sides[:, 3], dim=-1))
    zoom = side / size
    K_crop = torch.zeros_like(K)
    K_crop[:, :2, :2] = K[:, :2, :2] * zoom[:, None, None]
    K_crop[:, :2, 2] = side / 2
    K_crop[:, 2, 2] = 1.0
    return K_crop, R, zoom


def _half_scale(level, device):
    factor = 1 / 2 ** level
    shift = (factor - 1.0) / 2.0
    return torch.tensor([[factor, 0.0, shift], [0.0, factor, shift], [0.0, 0.0, 1.0]],
                        dtype=torch.float32, device=device)


def perspective_crops(images, K, K_crop, R, zoom, image_ids, side=SIDE):
    """Warp (C, 3, H, W) RGB images in [0, 1] into one (B, 3, side, side) crop per box.

    Sampled in linear light from a box-filtered pyramid, outside pixels white.
    """
    linear = images.pow(2.2).neg_().add_(1)
    levels = [linear]
    for _ in range(1, PYRAMID_LEVELS):
        levels.append(F.avg_pool2d(levels[-1], 2, 2))
    level_of = torch.clip(torch.floor(-torch.log2(zoom)), 0, PYRAMID_LEVELS - 1).int().tolist()

    device = images.device
    grid = torch.stack(torch.meshgrid(torch.arange(side, device=device),
                                      torch.arange(side, device=device), indexing="xy"), -1)
    grid = F.pad(grid.float(), (0, 1), value=1.0)                       # (side, side, 3)
    back = torch.linalg.inv(K_crop @ R)
    crops = []
    for i, level in enumerate(level_of):
        image = levels[level][image_ids[i]]
        rays = grid @ back[i].T
        rays = F.pad(rays[..., :2] / rays[..., 2:], (0, 1), value=1.0)
        pixels = (rays @ (_half_scale(level, device) @ K[i]).T)[..., :2]
        size = torch.tensor([image.shape[2], image.shape[1]], dtype=pixels.dtype, device=device)
        pixels = pixels.mul_(2.0 / (size - 1)).sub_(1.0)
        crops.append(F.grid_sample(image[None], pixels[None].to(image.dtype), mode="bilinear",
                                   padding_mode="zeros", align_corners=True))
    crops = torch.cat(crops).neg_().add_(1).clamp_(0, 1)
    return crops.pow_(GAMMA / 2.2)


def to_camera_frame(points, R, K):
    """Crop-frame (B, P, 3) landmarks back in each view's camera frame, with their pixels."""
    poses3d = (points.double() @ R.double()).float()
    pixels = F.pad(poses3d[..., :2] / poses3d[..., 2:], (0, 1), value=1.0)
    return poses3d, pixels @ K[:, :2, :].transpose(1, 2)
