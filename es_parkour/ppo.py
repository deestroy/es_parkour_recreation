"""PPO with GAE and KL-adaptive learning rate (rsl_rl-style schedule)."""
from __future__ import annotations

import numpy as np
import torch

from .config import PPOCfg


class RolloutBuffer:
    def __init__(self, horizon, num_envs, obs_dims, device):
        self.h, self.n = horizon, num_envs
        self.device = device
        self.obs = {k: torch.zeros(horizon, num_envs, d, device=device)
                    for k, d in obs_dims.items()}
        self.actions = torch.zeros(horizon, num_envs, 12, device=device)
        self.logp = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.step = 0

    def add(self, obs, action, logp, reward, done, value):
        i = self.step
        for k in self.obs:
            self.obs[k][i] = obs[k]
        self.actions[i] = action
        self.logp[i] = logp
        self.rewards[i] = torch.as_tensor(reward, device=self.device)
        self.dones[i] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[i] = value
        self.step += 1

    def compute_returns(self, last_value, gamma, lam):
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n, device=self.device)
        for t in reversed(range(self.h)):
            nonterminal = 1.0 - self.dones[t]
            next_v = last_value if t == self.h - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_v * nonterminal - self.values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        returns = adv + self.values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages, self.returns = adv, returns
        self.step = 0


class PPO:
    def __init__(self, policy, cfg: PPOCfg, device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.lr = cfg.lr

    def update(self, buf: RolloutBuffer):
        cfg = self.cfg
        b = buf.h * buf.n
        obs = {k: v.reshape(b, -1) for k, v in buf.obs.items()}
        actions = buf.actions.reshape(b, -1)
        old_logp = buf.logp.reshape(b)
        adv = buf.advantages.reshape(b)
        returns = buf.returns.reshape(b)
        old_values = buf.values.reshape(b)

        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
        count = 0
        mb = b // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(b, device=self.device)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mo = {k: v[idx] for k, v in obs.items()}
                logp, entropy, value, mean, std = self.policy.evaluate(mo, actions[idx])
                ratio = (logp - old_logp[idx]).exp()

                with torch.no_grad():
                    kl = (old_logp[idx] - logp).mean()
                    # rsl_rl-style adaptive lr
                    if kl > cfg.desired_kl * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < cfg.desired_kl / 2.0 and kl > 0.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.lr

                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()

                v_clipped = old_values[idx] + torch.clamp(
                    value - old_values[idx], -cfg.clip, cfg.clip)
                v_loss = torch.max((value - returns[idx]) ** 2,
                                   (v_clipped - returns[idx]) ** 2).mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += kl.item()
                stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                count += 1
        return {k: v / count for k, v in stats.items()} | {"lr": self.lr}
