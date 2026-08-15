"""Teacher ANN (paper Fig. 3, top row): scandot encoder -> depth latent,
privileged encoder, and the 3-layer MLP actor [512, 256, 128]."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import NetCfg, ObsCfg


def mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class TeacherActorCritic(nn.Module):
    def __init__(self, obs_cfg: ObsCfg, net_cfg: NetCfg, num_actions: int = 12,
                 init_noise_std: float = 1.0):
        super().__init__()
        o, n = obs_cfg, net_cfg
        self.scan_encoder = mlp([o.n_scan, *n.scan_hidden, n.scan_latent], out_act=nn.ELU)
        self.priv_encoder = mlp([o.n_priv, 64, n.priv_latent], out_act=nn.ELU)
        # critic owns separate encoders so its value loss cannot drag the
        # actor's representation (keeps the policy KL well-behaved)
        self.c_scan_encoder = mlp([o.n_scan, *n.scan_hidden, n.scan_latent], out_act=nn.ELU)
        self.c_priv_encoder = mlp([o.n_priv, 64, n.priv_latent], out_act=nn.ELU)
        in_dim = o.n_proprio + o.n_heading + n.scan_latent + n.priv_latent
        self.actor = mlp([in_dim, *n.actor_hidden, num_actions])
        self.critic = mlp([in_dim, *n.critic_hidden, 1])
        self.log_std = nn.Parameter(torch.full((num_actions,), float(torch.log(torch.tensor(init_noise_std)))))

    def encode(self, obs):
        scan_latent = self.scan_encoder(obs["scan"])
        priv_latent = self.priv_encoder(obs["priv"])
        return torch.cat([obs["proprio"], obs["heading"], scan_latent, priv_latent], dim=-1)

    def encode_critic(self, obs):
        scan_latent = self.c_scan_encoder(obs["scan"])
        priv_latent = self.c_priv_encoder(obs["priv"])
        return torch.cat([obs["proprio"], obs["heading"], scan_latent, priv_latent], dim=-1)

    def act(self, obs):
        mean = self.actor(self.encode(obs))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(self.encode_critic(obs)).squeeze(-1)
        return action, logp, value, mean, std

    def evaluate(self, obs, actions):
        mean = self.actor(self.encode(obs))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(self.encode_critic(obs)).squeeze(-1)
        return logp, entropy, value, mean, std

    @torch.no_grad()
    def act_inference(self, obs):
        return self.actor(self.encode(obs))


class RunningMeanStd:
    """Per-key observation normalizer (numpy-side, saved with checkpoints)."""

    def __init__(self, shape, clip=10.0):
        import numpy as np
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        import numpy as np
        bmean, bvar, bcount = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean += delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta ** 2 * self.count * bcount / tot) / tot
        self.count = tot

    def normalize(self, x):
        import numpy as np
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean, self.var, self.count = sd["mean"], sd["var"], sd["count"]
