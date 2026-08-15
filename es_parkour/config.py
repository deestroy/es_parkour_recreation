"""Central configuration for the ES-Parkour recreation.

Values follow the paper where stated (32 envs, lr 1e-3, actor MLP [512,256,128],
spiking timestep T=4, 10 Hz event sampling) and extreme-parkour conventions
elsewhere. IsaacGym is replaced by MuJoCo (CPU physics) + ROCm PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TerrainCfg:
    resolution: float = 0.05          # meters per heightfield cell
    course_length: float = 18.0       # meters (x)
    course_width: float = 4.0         # meters (y)
    platform_length: float = 2.0      # flat start platform
    num_obstacles: int = 8            # obstacles / waypoints per course
    obstacle_spacing: float = 1.8     # meters between obstacle starts
    pit_depth: float = 0.8            # depth of gaps
    # difficulty in [0,1] scales these ranges (low, high):
    gap_width: Tuple[float, float] = (0.10, 0.80)
    step_height: Tuple[float, float] = (0.05, 0.38)
    hurdle_height: Tuple[float, float] = (0.05, 0.40)
    hurdle_thickness: float = 0.12
    roughness: float = 0.02           # uniform noise added off-platform
    types: Tuple[str, ...] = ("gap", "step", "hurdle", "parkour")


@dataclass
class ObsCfg:
    # scandot grid in base-yaw frame
    scan_x: Tuple[float, float, int] = (-0.2, 1.4, 17)
    scan_y: Tuple[float, float, int] = (-0.5, 0.5, 11)
    n_scan: int = 17 * 11             # 187
    n_proprio: int = 46               # angvel3 gravity3 dof12 dofvel12 act12 contacts4
    n_heading: int = 3                # sin/cos yaw error + commanded speed
    n_priv: int = 5                   # base lin vel(3), friction, payload
    scan_clip: float = 1.0
    angvel_scale: float = 0.25
    dofvel_scale: float = 0.05


@dataclass
class ControlCfg:
    kp: float = 30.0
    kd: float = 0.8
    action_scale: float = 0.25
    action_clip: float = 6.0
    decimation: int = 4               # 50 Hz control at dt=0.005
    sim_dt: float = 0.005
    torque_limit_scale: float = 0.95


@dataclass
class RewardCfg:
    cmd_speed: float = 0.8            # m/s toward next waypoint
    tracking_goal_vel: float = 1.5
    tracking_yaw: float = 0.5
    waypoint_bonus: float = 0.5       # one-time, not dt-scaled
    lin_vel_z: float = -1.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.4
    torques: float = -1.0e-5
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    collision: float = -1.0
    feet_edge: float = -0.5
    termination: float = -1.0         # one-time, not dt-scaled


@dataclass
class EnvCfg:
    num_envs: int = 32                # paper: 32 parallel environments
    episode_length_s: float = 20.0
    spawn_x: float = 1.0
    spawn_z_offset: float = 0.32
    friction_range: Tuple[float, float] = (0.4, 1.25)
    payload_range: Tuple[float, float] = (-0.5, 1.5)   # kg added to base
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    # curriculum
    max_level: int = 9
    promote_frac: float = 0.875       # reach >= 7/8 waypoints
    demote_frac: float = 0.375


@dataclass
class DepthCamCfg:
    width: int = 64
    height: int = 64
    fov_deg: float = 87.0             # horizontal, D435-like
    near: float = 0.05
    far: float = 4.0
    pos: Tuple[float, float, float] = (0.32, 0.0, 0.03)  # on head, base frame
    pitch_deg: float = 25.0           # tilted down
    hz: float = 10.0                  # paper: events sampled at 10 Hz


@dataclass
class EventCfg:
    threshold_c: float = 0.10         # contrast threshold C (Eq. 1)
    max_events_per_px: int = 8
    log_eps: float = 1e-3


@dataclass
class PPOCfg:
    horizon: int = 64
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1.0e-3                # paper: 0.001
    desired_kl: float = 0.02
    entropy_coef: float = 0.005
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_noise_std: float = 1.0
    max_iterations: int = 20000
    save_every: int = 200


@dataclass
class NetCfg:
    scan_latent: int = 32
    priv_latent: int = 8
    scan_hidden: Tuple[int, int] = (256, 128)
    actor_hidden: Tuple[int, int, int] = (512, 256, 128)   # paper Sec III.A
    critic_hidden: Tuple[int, int, int] = (512, 256, 128)


@dataclass
class SNNCfg:
    timesteps: int = 4                # paper: spiking timestep 4
    event_latent: int = 32
    gru_hidden: int = 128
    actor_hidden: Tuple[int, int, int] = (512, 256, 128)   # spiking MLP actor
    lr: float = 1.0e-3
    warmup_iters: int = 2000          # supervised on teacher rollouts
    dagger_iters: int = 10000
    batch_size: int = 256
