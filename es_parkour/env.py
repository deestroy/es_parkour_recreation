"""Single Go2 parkour environment on a procedural heightfield course.

Observations (teacher):
  proprio (46): ang vel (3), projected gravity (3), joint pos offsets (12),
                joint vel (12), previous action (12), foot contacts (4)
  heading (3):  sin/cos of yaw error to active waypoint, commanded speed
  scandots (187): terrain height samples around the base, yaw-aligned grid
  priv (5):     base linear velocity (3), terrain friction, payload mass

Rewards follow extreme-parkour-style shaping with waypoint progress.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from .config import DepthCamCfg, EnvCfg, EventCfg
from .depth import DepthCamera
from .event_sim import EventSimulator
from .scene import build_scene_xml
from .terrain import Course

_HOME_QPOS = np.array([0, 0, 0.27, 1, 0, 0, 0,
                       0, 0.9, -1.8, 0, 0.9, -1.8,
                       0, 0.9, -1.8, 0, 0.9, -1.8], dtype=np.float64)
_DEFAULT_DOF = _HOME_QPOS[7:]


def quat_to_mat(q):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class ParkourEnv:
    def __init__(self, cfg: EnvCfg, assets_dir: str, seed: int = 0,
                 enable_vision: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.enable_vision = enable_vision
        if enable_vision:
            self.cam_cfg = DepthCamCfg()
            self.camera = DepthCamera(self.cam_cfg)
            self.event_sim = EventSimulator(EventCfg(), far=self.cam_cfg.far)
            self.last_events = np.zeros(
                (2, self.cam_cfg.height, self.cam_cfg.width), dtype=np.float32)
            self.last_depth = None
        t = cfg.terrain
        nrow = int(round(t.course_width / t.resolution)) + 1
        ncol = int(round(t.course_length / t.resolution)) + 1
        xml = build_scene_xml(assets_dir, nrow, ncol, t.course_length,
                              t.course_width, cfg.control.sim_dt)
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.terrain_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.terrain_hfield = m.geom_dataid[self.terrain_geom]
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

        # actuated joints: 12 motors in go2.xml order (FL FR RL RR x hip/thigh/calf)
        self.nu = m.nu
        assert self.nu == 12, f"expected 12 actuators, got {self.nu}"
        self.torque_limits = m.actuator_ctrlrange[:, 1].copy() * cfg.control.torque_limit_scale

        # classify robot collision geoms: feet (spheres on calves) vs the rest
        self.foot_geoms, self.penalty_geoms = [], []
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            if bid == 0:
                continue  # world/terrain
            if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
                continue  # visual only
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.endswith("_calf") and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.foot_geoms.append(gid)
            else:
                self.penalty_geoms.append(gid)
        assert len(self.foot_geoms) == 4, f"feet found: {len(self.foot_geoms)}"
        # order feet FL FR RL RR by body name for stable contact obs
        order = ["FL", "FR", "RL", "RR"]
        self.foot_geoms.sort(key=lambda g: order.index(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])[:2]))
        self.base_mass0 = float(m.body_mass[self.base_body])

        # scandot grid (base-yaw frame offsets)
        o = cfg.obs
        gx = np.linspace(o.scan_x[0], o.scan_x[1], o.scan_x[2])
        gy = np.linspace(o.scan_y[0], o.scan_y[1], o.scan_y[2])
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        self.scan_offsets = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (187,2)

        self.max_steps = int(cfg.episode_length_s * 50)
        self.course: Course = None
        self.level = 0
        self.terrain_type = "gap"
        self._ep_reward_sums = {}
        self.reset()

    # ------------------------------------------------------------------
    def set_task(self, terrain_type: str, level: int):
        self.terrain_type = terrain_type
        self.level = int(np.clip(level, 0, self.cfg.max_level))

    def reset(self):
        cfg = self.cfg
        difficulty = self.level / cfg.max_level
        self.course = Course(cfg.terrain, self.terrain_type, difficulty, self.rng)

        # upload heightfield: data in [0,1]; scale via hfield size elevation
        m = self.model
        hid = self.terrain_hfield
        m.hfield_data[:] = self.course.normalized_data().ravel()
        m.hfield_size[hid][2] = self.course.hmax - self.course.hmin
        m.hfield_size[hid][3] = 0.5 + max(0.0, -self.course.hmin)
        m.geom_pos[self.terrain_geom][2] = self.course.hmin

        # domain randomization
        self.friction = self.rng.uniform(*cfg.friction_range)
        m.geom_friction[self.terrain_geom][0] = self.friction
        self.payload = self.rng.uniform(*cfg.payload_range)
        m.body_mass[self.base_body] = self.base_mass0 + self.payload

        mujoco.mj_resetData(m, self.data)
        d = self.data
        d.qpos[:] = _HOME_QPOS
        d.qpos[0] = cfg.spawn_x
        d.qpos[1] = self.rng.uniform(-0.3, 0.3)
        ground = float(self.course.sample_height(d.qpos[0], d.qpos[1]))
        d.qpos[2] = ground + cfg.spawn_z_offset
        yaw = self.rng.uniform(-0.15, 0.15)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)

        self.step_count = 0
        self.active_wp = 0
        self.n_wp_reached = 0
        self.last_action = np.zeros(12)
        self.prev_action = np.zeros(12)
        self.last_dof_vel = np.zeros(12)
        self.finished = False
        self._ep_reward_sums = {}
        if self.enable_vision:
            self.event_sim.reset()
            self.last_events[:] = 0.0
            self.last_depth = None
        return self._observe()

    # ------------------------------------------------------------------
    def _target_waypoint(self):
        wps = self.course.waypoints
        if self.active_wp < len(wps):
            return wps[self.active_wp]
        return np.array([self.course.final_x + 1.0, 0.0])

    def step(self, action: np.ndarray, vision_tick: bool = False):
        cfg, m, d = self.cfg, self.model, self.data
        c = cfg.control
        action = np.clip(action, -c.action_clip, c.action_clip)
        self.prev_action = self.last_action
        self.last_action = action.copy()
        target = _DEFAULT_DOF + c.action_scale * action
        self.last_dof_vel = d.qvel[6:18].copy()

        for _ in range(c.decimation):
            tau = c.kp * (target - d.qpos[7:19]) - c.kd * d.qvel[6:18]
            d.ctrl[:] = np.clip(tau, -self.torque_limits, self.torque_limits)
            mujoco.mj_step(m, d)

        self.step_count += 1

        # waypoint progress
        base_xy = d.qpos[0:2]
        wp = self._target_waypoint()
        reached_bonus = 0.0
        if np.linalg.norm(base_xy - wp) < 0.6 and self.active_wp < len(self.course.waypoints):
            self.active_wp += 1
            self.n_wp_reached += 1
            reached_bonus = 1.0
        if d.qpos[0] > self.course.final_x:
            self.finished = True

        if self.enable_vision and vision_tick:
            R = quat_to_mat(d.qpos[3:7])
            depth = self.camera.render(m, d, d.qpos[0:3].copy(), R)
            self.last_events = self.event_sim.step(depth)
            self.last_depth = depth

        obs = self._observe()
        reward, rew_terms = self._reward(reached_bonus)
        term, fail = self._termination()
        timeout = self.step_count >= self.max_steps
        done = term or timeout or self.finished
        if fail:
            reward += cfg.reward.termination
        for k, v in rew_terms.items():
            self._ep_reward_sums[k] = self._ep_reward_sums.get(k, 0.0) + v

        info = {}
        if done:
            info = {
                "wp_frac": self.n_wp_reached / len(self.course.waypoints),
                "finished": self.finished,
                "fail": fail,
                "timeout": timeout and not term,
                "level": self.level,
                "type": self.terrain_type,
                "ep_len": self.step_count,
                "rew_terms": dict(self._ep_reward_sums),
            }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _base_frames(self):
        d = self.data
        R = quat_to_mat(d.qpos[3:7])
        lin_vel_b = R.T @ d.qvel[0:3]
        ang_vel_b = d.qvel[3:6]          # free joint: angular vel is body-frame
        gravity_b = R.T @ np.array([0.0, 0.0, -1.0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return R, lin_vel_b, ang_vel_b, gravity_b, yaw

    def _foot_state(self):
        d = self.data
        contacts = np.zeros(4)
        penalty_contacts = 0
        foot_pos = self.data.geom_xpos[self.foot_geoms]
        fset = set(self.foot_geoms)
        pset = set(self.penalty_geoms)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            for g in (g1, g2):
                if g in fset:
                    contacts[self.foot_geoms.index(g)] = 1.0
                elif g in pset:
                    penalty_contacts += 1
        return contacts, penalty_contacts, foot_pos

    def _observe(self):
        cfg, d = self.cfg, self.data
        o = cfg.obs
        R, lin_vel_b, ang_vel_b, gravity_b, yaw = self._base_frames()
        contacts, _, _ = self._foot_state()

        proprio = np.concatenate([
            ang_vel_b * o.angvel_scale,
            gravity_b,
            d.qpos[7:19] - _DEFAULT_DOF,
            d.qvel[6:18] * o.dofvel_scale,
            self.last_action,
            contacts - 0.5,
        ])

        wp = self._target_waypoint()
        to_wp = wp - d.qpos[0:2]
        yaw_err = np.arctan2(to_wp[1], to_wp[0]) - yaw
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
        heading = np.array([np.sin(yaw_err), np.cos(yaw_err), cfg.reward.cmd_speed])

        # scandots: rotate grid by yaw, sample terrain, express relative to base z
        cy, sy = np.cos(yaw), np.sin(yaw)
        px = d.qpos[0] + self.scan_offsets[:, 0] * cy - self.scan_offsets[:, 1] * sy
        py = d.qpos[1] + self.scan_offsets[:, 0] * sy + self.scan_offsets[:, 1] * cy
        hs = self.course.sample_height(px, py)
        scan = np.clip(d.qpos[2] - 0.27 - hs, -o.scan_clip, o.scan_clip)

        priv = np.concatenate([lin_vel_b, [self.friction, self.payload]])
        self._yaw_err = yaw_err
        out = {
            "proprio": proprio.astype(np.float32),
            "heading": heading.astype(np.float32),
            "scan": scan.astype(np.float32),
            "priv": priv.astype(np.float32),
        }
        if self.enable_vision:
            out["events"] = self.last_events
        return out

    def _reward(self, reached_bonus):
        cfg, d = self.cfg, self.data
        rw = cfg.reward
        R, lin_vel_b, ang_vel_b, gravity_b, yaw = self._base_frames()
        contacts, penalty_contacts, foot_pos = self._foot_state()

        wp = self._target_waypoint()
        to_wp = wp - d.qpos[0:2]
        dist = np.linalg.norm(to_wp)
        dir_wp = to_wp / (dist + 1e-6)
        v_along = float(d.qvel[0:2] @ dir_wp)

        tau = d.ctrl
        dof_acc = (d.qvel[6:18] - self.last_dof_vel) / (cfg.control.sim_dt * cfg.control.decimation)
        act_rate = self.last_action - self.prev_action
        feet_on_edge = self.course.near_edge(foot_pos[:, 0], foot_pos[:, 1])
        near_edge = float(np.sum(feet_on_edge * contacts))

        terms = {
            "goal_vel": rw.tracking_goal_vel * min(v_along, rw.cmd_speed) / rw.cmd_speed,
            "yaw": rw.tracking_yaw * float(np.exp(-np.abs(self._yaw_err))),
            "waypoint": rw.waypoint_bonus * reached_bonus,
            "lin_vel_z": rw.lin_vel_z * float(d.qvel[2] ** 2),
            "ang_vel_xy": rw.ang_vel_xy * float(np.sum(ang_vel_b[:2] ** 2)),
            "orientation": rw.orientation * float(np.sum(gravity_b[:2] ** 2)),
            "torques": rw.torques * float(np.sum(np.square(tau))),
            "dof_acc": rw.dof_acc * float(np.sum(np.square(dof_acc))),
            "action_rate": rw.action_rate * float(np.sum(np.square(act_rate))),
            "collision": rw.collision * float(penalty_contacts > 0),
            "feet_edge": rw.feet_edge * near_edge,
        }
        return float(sum(terms.values())), terms

    def _termination(self):
        d = self.data
        _, _, _, gravity_b, _ = self._base_frames()
        tilted = abs(gravity_b[0]) > 0.9 or abs(gravity_b[1]) > 0.9
        ground = float(self.course.sample_height(d.qpos[0], d.qpos[1]))
        in_pit = d.qpos[2] < ground - 0.05 and ground < -0.3
        fell = d.qpos[2] < self.course.hmin - 0.3
        off_side = abs(d.qpos[1]) > self.cfg.terrain.course_width / 2 - 0.15
        off_back = d.qpos[0] < 0.2
        fail = bool(tilted or in_pit or fell or off_side or off_back)
        return fail, fail
