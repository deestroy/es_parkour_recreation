"""Event camera simulation from depth frames (paper Sec. II-A, Fig. 4).

The paper generates events from depth images: brightness change
DeltaL(u) = L(u,t) - L(u,t-dt) (Eq. 2), realised through optical flow x
image gradient (Eq. 4). Between two sampled frames both formulations reduce
to the per-pixel log-intensity difference, which we threshold with contrast C
(Eq. 1) to emit signed events - the standard frame-difference event simulator.

Intensity is taken as inverse depth (near = bright), so the resulting events
are invariant to scene lighting by construction, matching the event camera's
high-dynamic-range behaviour the paper exploits (Table I).
"""
from __future__ import annotations

import numpy as np

from .config import EventCfg


class EventSimulator:
    """Per-environment stateful converter: depth frame -> 2-channel event frame."""

    def __init__(self, cfg: EventCfg, far: float):
        self.cfg = cfg
        self.far = far
        self.prev_logI = None

    def reset(self):
        self.prev_logI = None

    def step(self, depth: np.ndarray) -> np.ndarray:
        """depth (H,W) meters -> events (2,H,W): [positive, negative] counts,
        normalized to [0,1] by max_events_per_px."""
        c = self.cfg
        intensity = 1.0 - depth / self.far          # inverse depth as brightness
        logI = np.log(intensity + c.log_eps)
        if self.prev_logI is None:
            self.prev_logI = logI
            return np.zeros((2, *depth.shape), dtype=np.float32)
        dL = logI - self.prev_logI                  # Eq. 2
        n = np.floor(np.abs(dL) / c.threshold_c)    # events per pixel (Eq. 1)
        n = np.minimum(n, c.max_events_per_px)
        pos = np.where(dL > 0, n, 0.0)
        neg = np.where(dL < 0, n, 0.0)
        # reference intensity moves by the emitted events only (residual carries over)
        self.prev_logI = self.prev_logI + np.sign(dL) * n * c.threshold_c
        return (np.stack([pos, neg]) / c.max_events_per_px).astype(np.float32)
