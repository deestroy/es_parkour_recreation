"""Evaluation: success rates (Fig. 5), operation counts (Table II),
theoretical energy (Table III), joint motor energy (Table IV).

Usage:
  python scripts/evaluate.py --teacher runs/teacher/ckpt_latest.pt \
      [--student runs/student/student_latest.pt] [--episodes 32] [--level 9] \
      [--out docs/results.md]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_parkour.config import EnvCfg, NetCfg, SNNCfg  # noqa: E402
from es_parkour.energy import (E_AC, E_MAC, SNNOpCounter, ann_energy,  # noqa: E402
                               count_ann_flops, efficiency_ratio)
from es_parkour.env import ParkourEnv  # noqa: E402
from es_parkour.networks import RunningMeanStd, TeacherActorCritic  # noqa: E402
from es_parkour.snn import SNNStudent  # noqa: E402


def load_teacher(path, env_cfg, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = TeacherActorCritic(env_cfg.obs, NetCfg()).to(device)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    norm = {k: RunningMeanStd(1) for k in ("proprio", "scan", "priv")}
    for k in norm:
        norm[k].load_state_dict(ck["norm"][k])
    return pol, norm


def rollout(env, policy_fn, on_reset=None, max_steps=1000):
    """Runs one episode; returns (success, wp_frac, motor_energy_J, steps)."""
    obs = env.reset()
    if on_reset:
        on_reset()
    energy = 0.0
    dt = env.cfg.control.sim_dt * env.cfg.control.decimation
    for t in range(max_steps):
        act, vision_tick = policy_fn(obs, t)
        obs, r, done, info = env.step(act, vision_tick=vision_tick)
        # mechanical joint power |tau . qdot|
        energy += float(np.sum(np.abs(env.data.ctrl * env.data.qvel[6:18]))) * dt
        if done:
            return info["finished"], info["wp_frac"], energy, t + 1
    return False, env.n_wp_reached / len(env.course.waypoints), energy, max_steps


def eval_success(env_cfg, assets, actor_fn_builder, level, episodes, types):
    results = {}
    for ttype in types:
        env = ParkourEnv(env_cfg, assets, seed=123, enable_vision=True)
        env.set_task(ttype, level)
        succ, wps, energies = [], [], []
        policy_fn, on_reset = actor_fn_builder(env)
        for ep in range(episodes):
            s, w, e, n = rollout(env, policy_fn, on_reset)
            succ.append(s)
            wps.append(w)
            energies.append(e)
        results[ttype] = {"success": float(np.mean(succ)),
                          "wp_frac": float(np.mean(wps)),
                          "motor_energy_J": float(np.mean(energies))}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", default=None)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--level", type=int, default=9)
    p.add_argument("--out", default="docs/results.md")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = EnvCfg()
    assets = os.path.join(os.path.dirname(__file__), "..", "assets", "go2")
    teacher, norm = load_teacher(args.teacher, env_cfg, device)
    types = list(env_cfg.terrain.types)

    def teacher_builder(env):
        def fn(obs, t):
            to = {k: torch.as_tensor(norm[k].normalize(obs[k])[None], device=device,
                                     dtype=torch.float32)
                  for k in ("proprio", "scan", "priv")}
            to["heading"] = torch.as_tensor(obs["heading"][None], device=device)
            a = teacher.act_inference(to)[0].cpu().numpy()
            return a, (t + 1) % 5 == 0
        return fn, None

    print("evaluating teacher...", flush=True)
    teacher_res = eval_success(env_cfg, assets, teacher_builder, args.level,
                               args.episodes, types)
    print(teacher_res, flush=True)

    student_res, ops_report = None, None
    if args.student:
        sck = torch.load(args.student, map_location=device, weights_only=False)
        snn_cfg = SNNCfg()
        student = SNNStudent(snn_cfg, n_proprio=env_cfg.obs.n_proprio).to(device)
        student.load_state_dict(sck["student"])
        student.eval()

        def student_builder(env):
            state = {"h": student.init_hidden(1, device),
                     "latent": torch.zeros(1, snn_cfg.event_latent, device=device),
                     "head": torch.zeros(1, 2, device=device)}

            def on_reset():
                state["h"] = student.init_hidden(1, device)
                state["latent"] = torch.zeros(1, snn_cfg.event_latent, device=device)
                state["head"] = torch.zeros(1, 2, device=device)

            @torch.no_grad()
            def fn(obs, t):
                pn = torch.as_tensor(norm["proprio"].normalize(obs["proprio"])[None],
                                     device=device, dtype=torch.float32)
                if t % 5 == 0:
                    ev = torch.as_tensor(obs["events"][None], device=device)
                    state["latent"], state["head"] = student.encode(ev)
                a, state["h"] = student.act(pn, state["latent"], state["head"], state["h"])
                return a[0].cpu().numpy(), (t + 1) % 5 == 0
            return fn, on_reset

        print("evaluating student...", flush=True)
        student_res = eval_success(env_cfg, assets, student_builder, args.level,
                                   args.episodes, types)
        print(student_res, flush=True)

        # ---- operation counts on real event data (Tables II/III) -------
        print("counting ops...", flush=True)
        env = ParkourEnv(env_cfg, assets, seed=7, enable_vision=True)
        env.set_task("parkour", args.level)
        pol_fn, on_rst = student_builder(env)
        frames = []
        obs = env.reset()
        for t in range(250):
            a, vt = pol_fn(obs, t)
            obs, _, done, _ = env.step(a, vision_tick=vt)
            if t % 5 == 0:
                frames.append(obs["events"].copy())
            if done:
                obs = env.reset()
                on_rst()
        frames_t = torch.as_tensor(np.stack(frames), device=device)

        counter = SNNOpCounter(student.encoder)
        with torch.no_grad():
            for f in frames_t:
                student.encode(f[None])
        n = len(frames_t)
        snn_flops, snn_sops = counter.totals()
        snn_flops, snn_sops = snn_flops / n, snn_sops / n
        counter.remove()

        # ANN twin of the encoder (plain ResNet-18, same shape) for FLOPs
        from torchvision.models import resnet18
        ann_enc = resnet18(num_classes=34)
        ann_enc.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        ann_enc.eval()
        ann_flops = count_ann_flops(ann_enc, torch.zeros(1, 2, 64, 64))

        # actor ops: spiking MLP on gru_hidden+2 input
        actor_counter = SNNOpCounter(student.actor)
        with torch.no_grad():
            for _ in range(50):
                student.actor(torch.randn(1, snn_cfg.gru_hidden + 2, device=device))
        act_flops, act_sops = actor_counter.totals()
        act_flops, act_sops = act_flops / 50, act_sops / 50
        actor_counter.remove()
        h = snn_cfg.actor_hidden
        ann_actor_flops = ((snn_cfg.gru_hidden + 2) * h[0] + h[0] * h[1]
                           + h[1] * h[2] + h[2] * 12)

        enc_params = sum(p.numel() for p in student.encoder.parameters()) / 1e6
        act_params = sum(p.numel() for p in student.actor.parameters()) / 1e6
        ops_report = {
            "encoder": {"snn_flops": snn_flops, "snn_sops": snn_sops,
                        "ann_flops": ann_flops,
                        "eff": efficiency_ratio(snn_flops, snn_sops, ann_flops),
                        "snn_energy_mJ": (E_MAC * snn_flops + E_AC * snn_sops) * 1e3,
                        "ann_energy_mJ": ann_energy(ann_flops) * 1e3,
                        "params_M": enc_params},
            "actor": {"snn_flops": act_flops, "snn_sops": act_sops,
                      "ann_flops": ann_actor_flops,
                      "eff": efficiency_ratio(act_flops, act_sops, ann_actor_flops),
                      "snn_energy_mJ": (E_MAC * act_flops + E_AC * act_sops) * 1e3,
                      "ann_energy_mJ": ann_energy(ann_actor_flops) * 1e3,
                      "params_M": act_params},
        }
        print(ops_report, flush=True)

    # ---- report ------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# ES-Parkour recreation - evaluation results\n\n")
        f.write(f"Level {args.level}, {args.episodes} episodes/terrain.\n\n")
        f.write("## Success rates (cf. paper Fig. 5)\n\n")
        f.write("| Terrain | Teacher ANN | Student SNN |\n|---|---|---|\n")
        for t in types:
            s = f"{student_res[t]['success']*100:.0f}%" if student_res else "-"
            f.write(f"| {t} | {teacher_res[t]['success']*100:.0f}% | {s} |\n")
        f.write("\n## Joint motor energy per episode (cf. Table IV)\n\n")
        f.write("| Terrain | Teacher (J) | Student (J) |\n|---|---|---|\n")
        for t in types:
            s = f"{student_res[t]['motor_energy_J']:.1f}" if student_res else "-"
            f.write(f"| {t} | {teacher_res[t]['motor_energy_J']:.1f} | {s} |\n")
        if ops_report:
            f.write("\n## Operations & theoretical energy (cf. Tables II-III)\n\n")
            f.write("| Module | SNN FLOPs | SNN SOPs | ANN FLOPs | OPs(SNN):OPs(ANN) "
                    "| SNN mJ | ANN mJ | Saving |\n|---|---|---|---|---|---|---|---|\n")
            for mod in ("encoder", "actor"):
                r = ops_report[mod]
                saving = 100 * (1 - r["snn_energy_mJ"] / r["ann_energy_mJ"])
                f.write(f"| {mod} ({r['params_M']:.2f}M) | {r['snn_flops']:.3g} "
                        f"| {r['snn_sops']:.3g} | {r['ann_flops']:.3g} "
                        f"| {r['eff']:.2f} : 1 | {r['snn_energy_mJ']:.4g} "
                        f"| {r['ann_energy_mJ']:.4g} | {saving:.1f}% |\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
