"""Procedural parkour courses as MuJoCo heightfields.

Each course: flat start platform followed by `num_obstacles` segments of one
terrain type (gap / step / hurdle) or a random mix ("parkour"), with a waypoint
at every segment center. Difficulty in [0, 1] scales obstacle dimensions
(paper Fig. 5 terrain suite).
"""
from __future__ import annotations

import numpy as np

from .config import TerrainCfg


def _lerp(rng_pair, d):
    lo, hi = rng_pair
    return lo + (hi - lo) * d


class Course:
    """Heightfield course with analytic height queries (no rendering needed)."""

    def __init__(self, cfg: TerrainCfg, terrain_type: str, difficulty: float,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.type = terrain_type
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        res = cfg.resolution
        self.ncol = int(round(cfg.course_length / res)) + 1
        self.nrow = int(round(cfg.course_width / res)) + 1
        self.x0, self.y0 = 0.0, -cfg.course_width / 2.0
        h = np.zeros((self.nrow, self.ncol), dtype=np.float64)

        xs = self.x0 + np.arange(self.ncol) * res  # per-column x coordinate

        waypoints = []
        for i in range(cfg.num_obstacles):
            seg_start = cfg.platform_length + i * cfg.obstacle_spacing
            seg_center = seg_start + cfg.obstacle_spacing / 2.0
            seg_type = self.type
            d = self.difficulty
            if self.type == "parkour":
                seg_type = rng.choice(["gap", "step", "hurdle"])
                d = np.clip(d * rng.uniform(0.7, 1.1), 0.0, 1.0)

            if seg_type == "gap":
                w = _lerp(cfg.gap_width, d)
                mask = np.abs(xs - seg_center) < w / 2.0
                h[:, mask] = -cfg.pit_depth
            elif seg_type == "step":
                hh = _lerp(cfg.step_height, d)
                mask = np.abs(xs - seg_center) < 0.45
                h[:, mask] += hh
            elif seg_type == "hurdle":
                hh = _lerp(cfg.hurdle_height, d)
                mask = np.abs(xs - seg_center) < cfg.hurdle_thickness / 2.0
                h[:, mask] += hh

            wy = rng.uniform(-0.2, 0.2) if self.type == "parkour" else 0.0
            waypoints.append((seg_center, wy))

        # small roughness off the start platform, on walkable (non-pit) cells
        if cfg.roughness > 0:
            noise = rng.uniform(0.0, cfg.roughness, size=h.shape)
            noise[:, xs < cfg.platform_length] = 0.0
            noise[h < -0.5 * cfg.pit_depth] = 0.0
            h += noise

        self.height = h
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.final_x = cfg.platform_length + cfg.num_obstacles * cfg.obstacle_spacing

        # edge mask: cells whose 3x3 neighborhood spans a large height change
        hmax = h.copy()
        hmin = h.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                sh = np.roll(np.roll(h, dr, axis=0), dc, axis=1)
                hmax = np.maximum(hmax, sh)
                hmin = np.minimum(hmin, sh)
        self.edge_mask = (hmax - hmin) > 0.15

        self.hmin = float(h.min())
        self.hmax = float(h.max())
        if self.hmax - self.hmin < 1e-6:
            self.hmax = self.hmin + 1e-3

    # -- queries ----------------------------------------------------------
    def _cell_idx(self, x, y):
        res = self.cfg.resolution
        c = np.clip((np.asarray(x) - self.x0) / res, 0, self.ncol - 1.001)
        r = np.clip((np.asarray(y) - self.y0) / res, 0, self.nrow - 1.001)
        return r, c

    def sample_height(self, x, y):
        """Bilinear height lookup; x, y arrays in world frame."""
        r, c = self._cell_idx(x, y)
        r0 = np.floor(r).astype(int)
        c0 = np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        h = self.height
        return ((h[r0, c0] * (1 - fr) + h[r0 + 1, c0] * fr) * (1 - fc)
                + (h[r0, c0 + 1] * (1 - fr) + h[r0 + 1, c0 + 1] * fr) * fc)

    def near_edge(self, x, y):
        r, c = self._cell_idx(x, y)
        return self.edge_mask[np.round(r).astype(int), np.round(c).astype(int)]

    def normalized_data(self):
        """Heightfield data in [0,1] for mujoco, row-major float32."""
        return ((self.height - self.hmin) / (self.hmax - self.hmin)).astype(np.float32)
