"""Phase 2: distill the teacher ANN into the SNN student (paper Sec. II-C).

Usage: python scripts/distill_snn.py --teacher runs/teacher/ckpt_latest.pt \
           --logdir runs/student
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_parkour.config import SNNCfg  # noqa: E402
from es_parkour.distill import Distiller  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--logdir", default="runs/student")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    snn_cfg = SNNCfg()
    assets = os.path.join(os.path.dirname(__file__), "..", "assets", "go2")
    dist = Distiller(args.teacher, assets, device=device, snn_cfg=snn_cfg,
                     seed=args.seed)

    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    batch_chunks = max(4, snn_cfg.batch_size // 25)   # chunks per grad step
    t0 = time.time()
    it = 0

    def log(phase, losses):
        writer.add_scalar(f"{phase}/loss_action", losses["action"], it)
        writer.add_scalar(f"{phase}/loss_yaw", losses["yaw"], it)
        if it % 50 == 0:
            infos = dist.env.pop_episode_infos()
            wp = np.mean([e["wp_frac"] for e in infos]) if infos else float("nan")
            print(f"[{phase}] it {it:6d} | L_act {losses['action']:.4f} "
                  f"| L_yaw {losses['yaw']:.4f} | wp {wp:.2f} "
                  f"| buf {len(dist.buffer)} | {time.time()-t0:,.0f}s", flush=True)
            if infos:
                writer.add_scalar(f"{phase}/wp_frac", wp, it)

    # ---- warm-up: teacher drives, student imitates --------------------
    print("warm-up phase", flush=True)
    while it < snn_cfg.warmup_iters:
        dist.collect(2, behavior="teacher")
        for _ in range(8):
            losses = dist.train_batch(batch_chunks)
            log("warmup", losses)
            it += 1
        if it % 500 < 8:
            dist.save(os.path.join(args.logdir, "student_latest.pt"), it)

    # ---- DAgger: student drives, teacher labels -----------------------
    print("dagger phase", flush=True)
    while it < snn_cfg.warmup_iters + snn_cfg.dagger_iters:
        dist.collect(2, behavior="student")
        for _ in range(8):
            losses = dist.train_batch(batch_chunks)
            log("dagger", losses)
            it += 1
        if it % 500 < 8:
            dist.save(os.path.join(args.logdir, "student_latest.pt"), it)
    dist.save(os.path.join(args.logdir, "student_final.pt"), it)
    print("done", flush=True)


if __name__ == "__main__":
    main()
