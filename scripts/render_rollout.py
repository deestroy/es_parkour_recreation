"""Render a policy rollout to MP4 (run locally; needs OpenGL).

Usage: python scripts/render_rollout.py --teacher ckpt.pt --type hurdle --level 6 \
           [--student student.pt] --out rollout.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_parkour.config import EnvCfg, SNNCfg  # noqa: E402
from es_parkour.env import ParkourEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", default=None)
    p.add_argument("--type", default="parkour")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--out", default="rollout.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", action="store_true",
                   help="side panel with the event stream (red/blue, Fig. 1 style)")
    args = p.parse_args()

    from scripts.evaluate import load_teacher  # reuse loader
    env_cfg = EnvCfg()
    assets = os.path.join(os.path.dirname(__file__), "..", "assets", "go2")
    device = "cpu"
    teacher, norm = load_teacher(args.teacher, env_cfg, device)

    env = ParkourEnv(env_cfg, assets, seed=args.seed, enable_vision=True)
    env.set_task(args.type, args.level)
    obs = env.reset()

    student = None
    if args.student:
        from es_parkour.snn import SNNStudent
        sck = torch.load(args.student, map_location=device, weights_only=False)
        snn_cfg = SNNCfg()
        student = SNNStudent(snn_cfg, n_proprio=env_cfg.obs.n_proprio)
        student.load_state_dict(sck["student"])
        student.eval()
        h = student.init_hidden(1, device)
        latent = torch.zeros(1, snn_cfg.event_latent)
        head = torch.zeros(1, 2)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.base_body
    cam.distance, cam.elevation, cam.azimuth = 2.5, -15, 120

    frames = []
    with torch.no_grad():
        for t in range(1000):
            if student is None:
                to = {k: torch.as_tensor(norm[k].normalize(obs[k])[None],
                                         dtype=torch.float32)
                      for k in ("proprio", "scan", "priv")}
                to["heading"] = torch.as_tensor(obs["heading"][None])
                act = teacher.act_inference(to)[0].numpy()
            else:
                pn = torch.as_tensor(norm["proprio"].normalize(obs["proprio"])[None],
                                     dtype=torch.float32)
                if t % 5 == 0:
                    latent, head = student.encode(
                        torch.as_tensor(obs["events"][None]))
                act_t, h = student.act(pn, latent, head, h)
                act = act_t[0].numpy()
            obs, r, done, info = env.step(act, vision_tick=(t + 1) % 5 == 0)
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            if args.events:
                ev = obs["events"]
                panel = np.zeros((64, 64, 3), dtype=np.uint8)
                panel[..., 0] = (ev[0] * 255).astype(np.uint8)   # positive: red
                panel[..., 2] = (ev[1] * 255).astype(np.uint8)   # negative: blue
                import cv2
                panel = cv2.resize(panel, (160, 160), interpolation=cv2.INTER_NEAREST)
                frame[10:170, -170:-10] = panel
            frames.append(frame)
            if done:
                print("episode done:", info)
                break

    import imageio
    imageio.mimsave(args.out, frames, fps=50)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
