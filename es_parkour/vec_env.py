"""Threaded vectorized wrapper. mj_step releases the GIL, so a thread pool
scales across the 16 CPU cores without process overhead."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import EnvCfg
from .env import ParkourEnv


class VecParkourEnv:
    def __init__(self, cfg: EnvCfg, assets_dir: str, seed: int = 0, workers: int = 16):
        self.cfg = cfg
        self.n = cfg.num_envs
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.envs = [ParkourEnv(cfg, assets_dir, seed=seed + i) for i in range(self.n)]
        # fixed terrain-type assignment, round-robin over the paper's four types
        types = cfg.terrain.types
        self.types = [types[i % len(types)] for i in range(self.n)]
        self.levels = np.zeros(self.n, dtype=int)
        for e, t in zip(self.envs, self.types):
            e.set_task(t, 0)
        self.episode_infos = []

    def reset_all(self):
        obs = list(self.pool.map(lambda e: e.reset(), self.envs))
        return self._stack(obs)

    def step(self, actions: np.ndarray):
        def _step(i):
            env = self.envs[i]
            obs, rew, done, info = env.step(actions[i])
            if done:
                # curriculum: promote/demote on episode outcome, then auto-reset
                frac = info["wp_frac"]
                if frac >= self.cfg.promote_frac or info["finished"]:
                    self.levels[i] = min(self.levels[i] + 1, self.cfg.max_level)
                elif frac < self.cfg.demote_frac:
                    self.levels[i] = max(self.levels[i] - 1, 0)
                env.set_task(self.types[i], self.levels[i])
                info["next_level"] = self.levels[i]
                obs = env.reset()
            return obs, rew, done, info

        results = list(self.pool.map(_step, range(self.n)))
        obs = self._stack([r[0] for r in results])
        rews = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        self.episode_infos.extend([i for i in infos if i])
        return obs, rews, dones, infos

    def pop_episode_infos(self):
        out, self.episode_infos = self.episode_infos, []
        return out

    @staticmethod
    def _stack(obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
