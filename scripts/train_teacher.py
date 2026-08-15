"""Phase 1: train the teacher ANN with PPO (paper Sec. II-C).

Usage: python scripts/train_teacher.py --logdir runs/teacher [--iters N] [--resume ckpt]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_parkour.config import EnvCfg, NetCfg, PPOCfg  # noqa: E402
from es_parkour.networks import RunningMeanStd, TeacherActorCritic  # noqa: E402
from es_parkour.ppo import PPO, RolloutBuffer  # noqa: E402
from es_parkour.vec_env import VecParkourEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="runs/teacher")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    env_cfg, net_cfg, ppo_cfg = EnvCfg(), NetCfg(), PPOCfg()
    iters = args.iters or ppo_cfg.max_iterations
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assets = os.path.join(os.path.dirname(__file__), "..", "assets", "go2")
    env = VecParkourEnv(env_cfg, assets, seed=args.seed)
    policy = TeacherActorCritic(env_cfg.obs, net_cfg,
                                init_noise_std=ppo_cfg.init_noise_std).to(device)
    algo = PPO(policy, ppo_cfg, device)

    o = env_cfg.obs
    norm = {"proprio": RunningMeanStd(o.n_proprio), "scan": RunningMeanStd(o.n_scan),
            "priv": RunningMeanStd(o.n_priv)}
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"])
        algo.optimizer.load_state_dict(ck["optimizer"])
        algo.lr = ck.get("lr", ppo_cfg.lr)
        for k in norm:
            norm[k].load_state_dict(ck["norm"][k])
        start_iter = ck["iter"] + 1
        print(f"resumed from {args.resume} at iter {start_iter}")

    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    def to_torch(obs):
        out = {}
        for k, v in obs.items():
            if k in norm:
                norm[k].update(v)
                v = norm[k].normalize(v)
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
        return out

    obs_dims = {"proprio": o.n_proprio, "heading": o.n_heading,
                "scan": o.n_scan, "priv": o.n_priv}
    buf = RolloutBuffer(ppo_cfg.horizon, env_cfg.num_envs, obs_dims, device)

    obs = to_torch(env.reset_all())
    ep_rew_buf = deque(maxlen=100)
    ep_stats = deque(maxlen=100)
    cur_ep_rew = np.zeros(env_cfg.num_envs)
    t0 = time.time()

    for it in range(start_iter, iters):
        with torch.no_grad():
            for _ in range(ppo_cfg.horizon):
                action, logp, value, mu, sigma = policy.act(obs)
                nobs, rew, done, infos = env.step(action.cpu().numpy())
                buf.add(obs, action, logp, rew, done, value, mu, sigma)
                cur_ep_rew += rew
                for i, dn in enumerate(done):
                    if dn:
                        ep_rew_buf.append(cur_ep_rew[i])
                        cur_ep_rew[i] = 0.0
                obs = to_torch(nobs)
            _, _, last_value, _, _ = policy.act(obs)
        buf.compute_returns(last_value, ppo_cfg.gamma, ppo_cfg.lam)
        stats = algo.update(buf)
        ep_stats.extend(env.pop_episode_infos())

        if it % 10 == 0:
            steps = (it + 1 - start_iter) * ppo_cfg.horizon * env_cfg.num_envs
            sps = steps / (time.time() - t0)
            mean_rew = np.mean(ep_rew_buf) if ep_rew_buf else 0.0
            wp = np.mean([e["wp_frac"] for e in ep_stats]) if ep_stats else 0.0
            lvl = np.mean(env.levels)
            fin = np.mean([e["finished"] for e in ep_stats]) if ep_stats else 0.0
            print(f"it {it:5d} | rew {mean_rew:7.2f} | wp {wp:.2f} | fin {fin:.2f} "
                  f"| lvl {lvl:.2f} | kl {stats['kl']:.4f} | lr {stats['lr']:.1e} "
                  f"| {sps:,.0f} sps", flush=True)
            writer.add_scalar("train/ep_reward", mean_rew, it)
            writer.add_scalar("train/wp_frac", wp, it)
            writer.add_scalar("train/finished", fin, it)
            writer.add_scalar("train/mean_level", lvl, it)
            writer.add_scalar("train/sps", sps, it)
            for k, v in stats.items():
                writer.add_scalar(f"ppo/{k}", v, it)
            for t in env_cfg.terrain.types:
                lv = [env.levels[i] for i in range(env.n) if env.types[i] == t]
                writer.add_scalar(f"level/{t}", np.mean(lv), it)
            if ep_stats:
                terms = {}
                for e in ep_stats:
                    for k, v in e.get("rew_terms", {}).items():
                        terms.setdefault(k, []).append(v)
                for k, v in terms.items():
                    writer.add_scalar(f"reward/{k}", np.mean(v), it)

        if it % ppo_cfg.save_every == 0 or it == iters - 1:
            ck = {"policy": policy.state_dict(),
                  "optimizer": algo.optimizer.state_dict(),
                  "norm": {k: n.state_dict() for k, n in norm.items()},
                  "iter": it, "lr": algo.lr,
                  "levels": env.levels.tolist()}
            torch.save(ck, os.path.join(args.logdir, "ckpt_latest.pt"))
            if it % (ppo_cfg.save_every * 5) == 0:
                torch.save(ck, os.path.join(args.logdir, f"ckpt_{it:06d}.pt"))


if __name__ == "__main__":
    main()
