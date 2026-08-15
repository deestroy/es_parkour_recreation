# ES-Parkour Recreation

Recreation of *"ES-Parkour: Advanced Robot Parkour with Bio-inspired Event
Camera and Spiking Neural Network"* (Zhang et al., ICME 2025): a Unitree Go2
quadruped learns parkour (gaps, steps, hurdles, mixed courses) from a
simulated event camera processed by a spiking neural network, distilled from a
privileged ANN teacher trained with RL.

## Key substitution: MuJoCo instead of IsaacGym

The paper trains in IsaacGym, which requires an NVIDIA GPU. This recreation
targets an AMD Instinct MI210 (ROCm), so:

- **Physics**: MuJoCo on CPU, 32 parallel environments in a thread pool
  (the paper also uses 32 envs).
- **Networks**: PyTorch ROCm on the AMD GPU.
- **Depth camera**: `mj_multiRay` raycasting - no OpenGL needed headless.

Everything else follows the paper:

| Paper element | Here |
|---|---|
| Event generation from depth, Eq. 1-4, 10 Hz | `es_parkour/event_sim.py`, `depth.py` |
| Teacher: scandots + oracle heading + privileged obs, PPO, lr 1e-3 | `networks.py`, `ppo.py`, `scripts/train_teacher.py` |
| Terrains: gap / step / hurdle / parkour + curriculum (Fig. 5) | `terrain.py` |
| Spiking ResNet-18 encoder (11.19M), IF neurons, T=4 | `snn.py` |
| GRU fusion + 3-layer spiking MLP actor [512, 256, 128] | `snn.py` |
| Distillation: L_action + L_yaw (Eq. 8-9), warm-up then DAgger | `distill.py`, `scripts/distill_snn.py` |
| FLOPs/SOPs, E_MAC = 4.6 pJ / E_AC = 0.9 pJ, Eq. 10-11 (Tables II-III) | `energy.py` |
| Success rates / joint motor energy (Fig. 5, Table IV) | `scripts/evaluate.py` |

## Running

```bash
# phase 1: teacher
python scripts/train_teacher.py --logdir runs/teacher

# phase 2: distillation to SNN
python scripts/distill_snn.py --teacher runs/teacher/ckpt_latest.pt --logdir runs/student

# evaluation + tables
python scripts/evaluate.py --teacher runs/teacher/ckpt_latest.pt \
    --student runs/student/student_final.pt --out docs/results.md

# video (local, needs OpenGL)
python scripts/render_rollout.py --teacher ckpt.pt --type hurdle --level 6 --events
```

`remote/sync.sh` pushes the repo to the GPU host (`claude_svpn`).

## Known deviations from the paper

- MuJoCo replaces IsaacGym (AMD GPU constraint); heightfield terrain replaces
  their scandot terrain meshes.
- Events come from per-pixel log-inverse-depth differences between consecutive
  10 Hz depth frames; the paper interpolates via optical flow x gradient
  (Eq. 4) - both discretize the same DeltaL and ours is exact for the sampled
  frames rather than a flow approximation.
- The GRU fuses at 50 Hz on the held 10 Hz event latent (the paper does not
  specify the fusion rate).
- Success-rate absolute numbers are not comparable 1:1 (different simulator,
  contact model and obstacle scaling), but relative ANN vs SNN trends and the
  energy analysis reproduce the paper's methodology.
