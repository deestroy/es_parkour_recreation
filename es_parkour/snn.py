"""SNN student (paper Fig. 3 bottom row + Sec. III-A).

- Spiking ResNet-18 encoder (IF neurons, ATan surrogate, T=4 timesteps) on
  2-channel event frames -> event latent (32) + predicted heading (sin, cos).
- GRU fuses proprioception with the event latent (paper: "GRU module to fuse
  the latent features encoded from proprioceptive information and event
  features").
- 3-layer spiking MLP actor [512, 256, 128] -> rate-decoded linear readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from .config import SNNCfg


def _if_node():
    return neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = layer.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = layer.BatchNorm2d(cout)
        self.sn1 = _if_node()
        self.conv2 = layer.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = layer.BatchNorm2d(cout)
        self.sn2 = _if_node()
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(layer.Conv2d(cin, cout, 1, stride, bias=False),
                                      layer.BatchNorm2d(cout))

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.sn2(out + identity)


class SpikingResNet18Encoder(nn.Module):
    """Input (T, B, 2, 64, 64) -> (latent (B, 32), heading (B, 2))."""

    def __init__(self, cfg: SNNCfg, in_ch: int = 2):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            layer.Conv2d(in_ch, 64, 7, 2, 3, bias=False),
            layer.BatchNorm2d(64), _if_node(),
            layer.MaxPool2d(3, 2, 1))
        blocks = []
        cin = 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1),
                             (256, 2), (256, 1), (512, 2), (512, 1)]:
            blocks.append(BasicBlock(cin, cout, stride))
            cin = cout
        self.layers = nn.Sequential(*blocks)
        self.pool = layer.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, cfg.event_latent + 2)
        functional.set_step_mode(self, "m")
        # keep the rate-decoded readout an ordinary Linear on time-averaged spikes
        self.head_latent_dim = cfg.event_latent

    def forward(self, x_seq):
        functional.reset_net(self)
        s = self.stem(x_seq)
        s = self.layers(s)
        s = self.pool(s)                      # (T, B, 512, 1, 1)
        rate = s.flatten(2).mean(0)           # rate decoding over T
        out = self.head(rate)
        latent, heading = out[:, :self.head_latent_dim], out[:, self.head_latent_dim:]
        return latent, heading


class SpikingActor(nn.Module):
    """(B, in_dim) repeated over T -> (B, 12) actions, rate-decoded."""

    def __init__(self, cfg: SNNCfg, in_dim: int, num_actions: int = 12):
        super().__init__()
        self.cfg = cfg
        h = cfg.actor_hidden
        self.net = nn.Sequential(
            layer.Linear(in_dim, h[0]), _if_node(),
            layer.Linear(h[0], h[1]), _if_node(),
            layer.Linear(h[1], h[2]), _if_node())
        self.readout = nn.Linear(h[2], num_actions)
        functional.set_step_mode(self.net, "m")

    def forward(self, x):
        functional.reset_net(self.net)
        x_seq = x.unsqueeze(0).repeat(self.cfg.timesteps, 1, 1)
        s = self.net(x_seq)
        return self.readout(s.mean(0))


class SNNStudent(nn.Module):
    """Full student: encoder (10 Hz) + GRU fusion + spiking actor (50 Hz)."""

    def __init__(self, cfg: SNNCfg, n_proprio: int = 46, num_actions: int = 12):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpikingResNet18Encoder(cfg)
        self.gru = nn.GRUCell(n_proprio + cfg.event_latent, cfg.gru_hidden)
        self.actor = SpikingActor(cfg, cfg.gru_hidden + 2, num_actions)

    def encode(self, events):
        """events (B, 2, H, W) -> latent (B, 32), heading (B, 2). Repeats the
        frame over T=4 spiking timesteps (direct coding)."""
        x_seq = events.unsqueeze(0).repeat(self.cfg.timesteps, 1, 1, 1, 1)
        return self.encoder(x_seq)

    def act(self, proprio, latent, heading, h):
        h = self.gru(torch.cat([proprio, latent], dim=-1), h)
        action = self.actor(torch.cat([h, heading], dim=-1))
        return action, h

    def init_hidden(self, batch, device):
        return torch.zeros(batch, self.cfg.gru_hidden, device=device)
