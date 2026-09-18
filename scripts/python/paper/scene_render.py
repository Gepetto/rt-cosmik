"""Headless screenshots of a meshcat scene, for the paper's 3D figures.

meshcat draws in a browser, so figures are taken from headless Chromium
(playwright, WebGL through SwiftShader). Two kinds of camera:

* :meth:`Renderer.look_at` -- a free perspective or orthographic view;
* :meth:`Renderer.calibrated` -- exactly one of COMFI's cameras: its pose and
  intrinsics become three.js' view and projection matrices, so the render lines
  up pixel for pixel with the (undistorted) video frame it is laid over.

meshcat's own ``render`` re-applies its orbit controls every frame, which would
undo both; it is replaced in the page by a plain draw. Screenshots keep an alpha
channel (meshcat's WebGL context has one) once the background is hidden.
"""
import time

import numpy as np

#: OpenCV camera axes (x right, y down, z forward) -> three.js (x right, y up, looking -z).
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

_FIX_CAMERA = """
(args) => {
  const v = viewer, cam = v.camera;
  v.controls.enabled = false;
  v.render = function () {
    this.renderer.render(this.scene, this.camera);
    this.animator.after_render();
    this.needs_render = false;
  };
  cam.matrixAutoUpdate = false;
  // meshcat's world is z up; its scene root turns it into three.js' y-up frame.
  v.scene.updateMatrixWorld(true);
  const world = v.scene.matrixWorld.clone().multiply(
      new cam.matrix.constructor().fromArray(args.world));                 // column-major
  const parentInv = cam.parent.matrixWorld.clone().invert();
  cam.matrix.copy(parentInv.multiply(world));
  cam.matrixWorldNeedsUpdate = true;
  cam.projectionMatrix.fromArray(args.projection);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  v.renderer.setClearColor(0xffffff, args.alpha ? 0 : 1);
  v.set_dirty();
  return true;
}
"""


def perspective(K, width, height, near=0.05, far=50.0):
    """OpenGL projection (column-major list) for a pinhole camera K, image size."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([[2 * fx / width, 0, 1 - 2 * cx / width, 0],
                  [0, 2 * fy / height, 2 * cy / height - 1, 0],
                  [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                  [0, 0, -1, 0]])
    return P.T.ravel().tolist()


def orthographic(half_width, half_height, near=0.05, far=50.0):
    P = np.array([[1 / half_width, 0, 0, 0], [0, 1 / half_height, 0, 0],
                  [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
    return P.T.ravel().tolist()


def look_at_cv(eye, target, up=(0.0, 0.0, 1.0)):
    """world_T_camera with OpenCV axes, looking from ``eye`` at ``target``."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:               # looking straight along up
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, y, z]), eye
    return T


class Renderer:
    """A meshcat viewer and a headless browser looking at it.

    ``size`` is the CSS viewport in pixels; ``scale`` multiplies the pixel
    density of the screenshot (anti-aliasing and print resolution).
    """

    def __init__(self, size=(1280, 720), scale=2):
        import meshcat
        from playwright.sync_api import sync_playwright
        self.vis = meshcat.Visualizer()
        self.size, self.scale = tuple(size), scale
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        self.page = self._browser.new_page(viewport={"width": size[0], "height": size[1]},
                                           device_scale_factor=scale)
        self.page.goto(self.vis.url())
        self.page.wait_for_function("() => typeof viewer !== 'undefined' && viewer.renderer")
        self.page.add_style_tag(content=".dg.ac, .dg.main { display: none !important; } "
                                        "body { background: transparent !important; }")
        self._camera = None

    def close(self):
        self._browser.close()
        self._pw.stop()
        server = getattr(self.vis.window, "server_proc", None)   # the zmq server meshcat spawned
        if server is not None:
            server.kill()
            server.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- cameras -----------------------------------------------------------

    def calibrated(self, world_T_cam, K, image_size):
        """Look through a calibrated camera; the viewport must have the image's aspect."""
        self._camera = (np.asarray(world_T_cam, float) @ CV_TO_GL,
                        perspective(K, image_size[0], image_size[1]))

    def look_at(self, eye, target, fov_deg=30.0, up=(0.0, 0.0, 1.0), ortho_half_height=None):
        """A free camera; ``ortho_half_height`` (m) makes it orthographic."""
        world = look_at_cv(eye, target, up) @ CV_TO_GL
        aspect = self.size[0] / self.size[1]
        if ortho_half_height is not None:
            projection = orthographic(ortho_half_height * aspect, ortho_half_height)
        else:
            f = 0.5 * self.size[1] / np.tan(np.radians(fov_deg) / 2)
            K = np.array([[f, 0, self.size[0] / 2], [0, f, self.size[1] / 2], [0, 0, 1]])
            projection = perspective(K, *self.size)
        self._camera = (world, projection)

    # -- capture -----------------------------------------------------------

    def settle(self, seconds=0.4):
        """Let queued meshcat commands land (they travel over a websocket)."""
        time.sleep(seconds)
        self.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")

    def shot(self, path=None, alpha=False, background=True, grid=True):
        """Screenshot as an (H, W, 4) uint8 RGBA array; optionally saved."""
        import io
        from PIL import Image
        self.vis["/Background"].set_property("visible", bool(background and not alpha))
        self.vis["/Grid"].set_property("visible", bool(grid))
        self.vis["/Axes"].set_property("visible", False)
        self.settle()
        if self._camera is not None:
            world, projection = self._camera
            self.page.evaluate(_FIX_CAMERA, {"world": world.T.ravel().tolist(),
                                             "projection": projection, "alpha": alpha})
        self.page.evaluate("() => { viewer.set_dirty(); viewer.render(); }")
        self.settle(0.1)
        png = self.page.screenshot(omit_background=alpha, type="png")
        image = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        if path is not None:
            Image.fromarray(image).save(path)
        return image


def crop_to_content(rgba, pad=8, background=(255, 255, 255)):
    """Tight crop around non-background (or non-transparent) pixels."""
    if rgba.shape[2] == 4 and rgba[..., 3].min() < 255:
        mask = rgba[..., 3] > 0
    else:
        mask = np.any(np.abs(rgba[..., :3].astype(int) - np.asarray(background)) > 6, axis=2)
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return rgba
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + pad + 1)
    return rgba[y0:y1, x0:x1]


def composite(image_rgb, overlay_rgba, opacity=1.0):
    """Alpha-blend an RGBA render over an RGB image of the same size."""
    a = overlay_rgba[..., 3:4].astype(float) / 255.0 * opacity
    out = image_rgb.astype(float) * (1 - a) + overlay_rgba[..., :3].astype(float) * a
    return out.clip(0, 255).astype(np.uint8)
