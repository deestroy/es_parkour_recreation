"""Head-mounted depth camera via mj_multiRay raycasting (paper Fig. 4 input).

No OpenGL needed: rays hit only terrain geoms (group 0), so it runs on the
headless GPU server. Returns radial depth in meters, shape (H, W).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config import DepthCamCfg


class DepthCamera:
    def __init__(self, cfg: DepthCamCfg):
        self.cfg = cfg
        h, w = cfg.height, cfg.width
        # pixel grid in camera frame: x forward, y left, z up
        tan_h = np.tan(np.radians(cfg.fov_deg) / 2.0)
        tan_v = tan_h * h / w
        u = np.linspace(tan_h, -tan_h, w)      # left -> right maps +y -> -y
        v = np.linspace(tan_v, -tan_v, h)      # top -> bottom maps +z -> -z
        vv, uu = np.meshgrid(v, u, indexing="ij")
        dirs = np.stack([np.ones_like(uu), uu, vv], axis=-1)   # (H,W,3)
        self.dirs_cam = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).reshape(-1, 3)
        p = np.radians(cfg.pitch_deg)
        self.R_pitch = np.array([[np.cos(p), 0, -np.sin(p)],
                                 [0, 1, 0],
                                 [np.sin(p), 0, np.cos(p)]])
        self.pos_b = np.array(cfg.pos)
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        self.geomgroup[0] = 1                  # terrain only
        n = h * w
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)
        self._vec = np.zeros(n * 3, dtype=np.float64)

    def render(self, model, data, base_pos, base_R):
        """Depth image from the current robot pose. base_R: 3x3 world rotation."""
        cfg = self.cfg
        origin = base_pos + base_R @ self.pos_b
        R_cam = base_R @ self.R_pitch
        dirs_w = self.dirs_cam @ R_cam.T
        self._vec[:] = dirs_w.ravel()
        args = [model, data, np.asarray(origin, dtype=np.float64),
                self._vec, self.geomgroup, 1, -1,
                self._geomid, self._dist]
        try:
            # mujoco >= 3.3 signature (extra optional surface-normal output)
            mujoco.mj_multiRay(*args, None, len(self.dirs_cam), cfg.far)
        except TypeError:
            mujoco.mj_multiRay(*args, len(self.dirs_cam), cfg.far)
        depth = self._dist.copy()
        depth[self._geomid == -1] = cfg.far
        return np.clip(depth, cfg.near, cfg.far).reshape(cfg.height, cfg.width)
