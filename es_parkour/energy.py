"""Operation counting and theoretical energy (paper Sec. II-D, Eq. 10-11).

FLOPs: MAC operations of dense (ANN) layers.
SOPs:  accumulate-only operations in spiking layers = spikes_in x fan-out,
       measured empirically by hooking the IF neurons on real event data.
Energy: E_MAC = 4.6 pJ, E_AC = 0.9 pJ (45 nm, paper's constants). Eq. 10:
       E = E_MAC * FLOP_conv1 + E_AC * (sum SOP_conv + sum SOP_fc).
Efficiency (Eq. 11): OPs(SNN) : OPs(ANN) = (FLOP_snn + SOP_snn) / FLOP_ann.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

E_MAC = 4.6e-12   # J
E_AC = 0.9e-12    # J


def _conv_flops(mod, x_shape, y_shape):
    # y: (B, Cout, H, W); MACs = Cout*H*W * Cin*k*k / groups
    cout, h, w = y_shape[1], y_shape[2], y_shape[3]
    cin = x_shape[1]
    k = mod.kernel_size[0] * mod.kernel_size[1]
    return cout * h * w * cin * k // mod.groups


def _linear_flops(mod):
    return mod.in_features * mod.out_features


def count_ann_flops(model: nn.Module, example: dict) -> int:
    """Total MACs of an ANN module for one forward pass."""
    flops = [0]
    hooks = []

    def conv_hook(m, i, o):
        flops[0] += _conv_flops(m, i[0].shape, o.shape)

    def lin_hook(m, i, o):
        flops[0] += _linear_flops(m) * (o.numel() // o.shape[-1])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(**example) if isinstance(example, dict) else model(example)
    for h in hooks:
        h.remove()
    return flops[0]


class SNNOpCounter:
    """Counts per-layer SOPs of a spiking model on real inputs.

    For each spiking Conv/Linear layer, SOP = (incoming spike count) x
    (synaptic fan-out per spike). The first Conv layer receives analog event
    frames, so it is counted as FLOPs (Eq. 10 treats it with E_MAC).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.reset()
        self.hooks = []
        self.layer_kind = {}
        prev_spiking = {"flag": False}   # first conv sees analog input

        for name, m in model.named_modules():
            if isinstance(m, (layer.Conv2d, nn.Conv2d)) and not isinstance(m, nn.Conv1d):
                kind = "conv_analog" if not prev_spiking["flag"] else "conv"
                prev_spiking["flag"] = True
                self.layer_kind[name] = kind
                self.hooks.append(m.register_forward_hook(self._make_hook(name, m, kind)))
            elif isinstance(m, (layer.Linear, nn.Linear)):
                self.layer_kind[name] = "fc"
                self.hooks.append(m.register_forward_hook(self._make_hook(name, m, "fc")))

    def _make_hook(self, name, mod, kind):
        def hook(m, inp, out):
            x = inp[0]
            batch = x.shape[1] if x.dim() >= 5 else (x.shape[0] if x.dim() >= 2 else 1)
            if kind == "conv_analog":
                # analog input conv: dense MACs (per sample, summed over T)
                t = x.shape[0] if x.dim() == 5 else 1
                y = out
                ops = _conv_flops(m, x.reshape(-1, *x.shape[-3:]).shape,
                                  y.reshape(-1, *y.shape[-3:]).shape) * t
                self.flops[name] = self.flops.get(name, 0) + ops
            else:
                spikes = float(x.sum().item())
                if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                    k = m.kernel_size[0] * m.kernel_size[1]
                    fan_out = m.out_channels * k / (m.stride[0] * m.stride[1]) ** 1
                else:
                    fan_out = m.out_features
                self.sops[name] = self.sops.get(name, 0) + spikes * fan_out / max(1, batch)
            self.samples = max(self.samples, 1)
        return hook

    def reset(self):
        self.flops = {}
        self.sops = {}
        self.samples = 0
        self.n_forwards = 0

    def totals(self):
        return sum(self.flops.values()), sum(self.sops.values())

    def energy(self):
        f, s = self.totals()
        return E_MAC * f + E_AC * s

    def remove(self):
        for h in self.hooks:
            h.remove()


def ann_energy(flops: int) -> float:
    return E_MAC * flops


def efficiency_ratio(snn_flops, snn_sops, ann_flops) -> float:
    """Eq. 11: OPs(SNN) : OPs(ANN); < 1 means the SNN is cheaper."""
    return (snn_flops + snn_sops) / ann_flops
