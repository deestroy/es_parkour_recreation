"""ANN-to-SNN distillation (paper Fig. 3, Eq. 8-9).

Warm-up phase: the frozen teacher acts; the student regresses teacher actions
(L_action) and the oracle heading (L_yaw) over rollout chunks.
DAgger phase: the student acts in the environment while the teacher labels the
visited states ("we continue to measure and minimize the loss under identical
environmental conditions").
"""
from __future__ import annotations

import os
import random
import time
from collections import deque

import numpy as np
import torch

from .config import EnvCfg, NetCfg, SNNCfg
from .networks import RunningMeanStd, TeacherActorCritic
from .snn import SNNStudent

CHUNK = 25            # control steps per training chunk (0.5 s, 5 event frames)
VISION_EVERY = 5


class ChunkBuffer:
    def __init__(self, capacity=4096):
        self.buf = deque(maxlen=capacity)

    def push(self, chunk):
        self.buf.append(chunk)

    def sample(self, k):
        return random.sample(self.buf, min(k, len(self.buf)))

    def __len__(self):
        return len(self.buf)


class Distiller:
    def __init__(self, teacher_ckpt: str, assets_dir: str, device="cuda",
                 env_cfg: EnvCfg = None, snn_cfg: SNNCfg = None, seed: int = 1):
        from .vec_env import VecParkourEnv
        self.env_cfg = env_cfg or EnvCfg()
        self.snn_cfg = snn_cfg or SNNCfg()
        self.device = device
        self.env = VecParkourEnv(self.env_cfg, assets_dir, seed=seed,
                                 enable_vision=True)

        ck = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        self.teacher = TeacherActorCritic(self.env_cfg.obs, NetCfg()).to(device)
        self.teacher.load_state_dict(ck["policy"])
        self.teacher.eval()
        o = self.env_cfg.obs
        self.norm = {"proprio": RunningMeanStd(o.n_proprio),
                     "scan": RunningMeanStd(o.n_scan),
                     "priv": RunningMeanStd(o.n_priv)}
        for k in self.norm:
            self.norm[k].load_state_dict(ck["norm"][k])
        # start every env at the teacher's trained curriculum levels
        levels = ck.get("levels")
        if levels:
            for i, e in enumerate(self.env.envs):
                self.env.levels[i] = levels[i % len(levels)]
                e.set_task(self.env.types[i], self.env.levels[i])

        self.student = SNNStudent(self.snn_cfg, n_proprio=o.n_proprio).to(device)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.snn_cfg.lr)
        self.buffer = ChunkBuffer()

    # ------------------------------------------------------------------
    def _teacher_obs(self, obs):
        out = {}
        for k in ("proprio", "scan", "priv"):
            out[k] = torch.as_tensor(self.norm[k].normalize(obs[k]),
                                     device=self.device)
        out["heading"] = torch.as_tensor(obs["heading"], device=self.device)
        return out

    @torch.no_grad()
    def collect(self, n_chunks: int, behavior: str = "teacher"):
        """Roll the env for n_chunks * CHUNK steps, pushing per-env chunks."""
        env, dev = self.env, self.device
        obs = env._stack([e._observe() for e in env.envs])
        h = self.student.init_hidden(env.n, dev)
        latent = torch.zeros(env.n, self.snn_cfg.event_latent, device=dev)
        head_pred = torch.zeros(env.n, 2, device=dev)

        for _ in range(n_chunks):
            chunk = {"h0": h.clone().cpu().numpy(),
                     "events": [], "proprio": [], "teacher_act": [],
                     "yaw": [], "dones": []}
            for t in range(CHUNK):
                tobs = self._teacher_obs(obs)
                teacher_act = self.teacher.act_inference(tobs)
                proprio_n = tobs["proprio"]

                if t % VISION_EVERY == 0:
                    ev = torch.as_tensor(obs["events"], device=dev)
                    latent, head_pred = self.student.encode(ev)
                    chunk["events"].append(obs["events"].copy())
                action, h = self.student.act(proprio_n, latent, head_pred, h)
                if behavior == "teacher":
                    step_act = teacher_act
                else:
                    step_act = action

                yaw_true = np.stack([obs["heading"][:, 0], obs["heading"][:, 1]], 1)
                chunk["proprio"].append(proprio_n.cpu().numpy())
                chunk["teacher_act"].append(teacher_act.cpu().numpy())
                chunk["yaw"].append(yaw_true)

                obs, _, done, _ = env.step(step_act.cpu().numpy())
                chunk["dones"].append(done.copy())
                if done.any():
                    idx = torch.as_tensor(np.nonzero(done)[0], device=dev)
                    h[idx] = 0.0
            # split the batched chunk into per-env samples
            for i in range(env.n):
                self.buffer.push({
                    "h0": chunk["h0"][i],
                    "events": np.stack([f[i] for f in chunk["events"]]),
                    "proprio": np.stack([p[i] for p in chunk["proprio"]]),
                    "teacher_act": np.stack([a[i] for a in chunk["teacher_act"]]),
                    "yaw": np.stack([y[i] for y in chunk["yaw"]]),
                    "dones": np.stack([d[i] for d in chunk["dones"]]),
                })
        return h

    # ------------------------------------------------------------------
    def train_batch(self, batch_size: int):
        """One gradient step on a batch of chunks. Returns loss dict."""
        samples = self.buffer.sample(batch_size)
        dev = self.device
        B = len(samples)
        events = torch.as_tensor(np.stack([s["events"] for s in samples]), device=dev)
        proprio = torch.as_tensor(np.stack([s["proprio"] for s in samples]), device=dev)
        t_act = torch.as_tensor(np.stack([s["teacher_act"] for s in samples]), device=dev)
        yaw = torch.as_tensor(np.stack([s["yaw"] for s in samples]), device=dev)
        dones = torch.as_tensor(np.stack([s["dones"] for s in samples]),
                                dtype=torch.float32, device=dev)
        h = torch.as_tensor(np.stack([s["h0"] for s in samples]), device=dev)

        n_frames = events.shape[1]
        loss_action = 0.0
        loss_yaw = 0.0
        latent, head_pred = None, None
        for t in range(CHUNK):
            if t % VISION_EVERY == 0:
                k = t // VISION_EVERY
                if k < n_frames:
                    latent, head_pred = self.student.encode(events[:, k])
                    loss_yaw = loss_yaw + torch.nn.functional.mse_loss(
                        head_pred, yaw[:, t])
            action, h = self.student.act(proprio[:, t], latent, head_pred, h)
            loss_action = loss_action + torch.nn.functional.mse_loss(action, t_act[:, t])
            h = h * (1.0 - dones[:, t]).unsqueeze(-1)

        loss_action = loss_action / CHUNK
        loss_yaw = loss_yaw / max(1, n_frames)
        loss = loss_action + loss_yaw          # Eq. 8 + Eq. 9
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.opt.step()
        return {"action": float(loss_action.detach()), "yaw": float(loss_yaw.detach())}

    def save(self, path, it):
        torch.save({"student": self.student.state_dict(),
                    "opt": self.opt.state_dict(),
                    "iter": it,
                    "snn_cfg": self.snn_cfg.__dict__}, path)
