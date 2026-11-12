# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_inv, quat_mul

from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
)
from wuji_mjlab.utils.math import (
  cube_rotation_quats,
  nearest_cube_rotation_index,
  random_quat_uniform,
)

_GOAL_MODES = ("uniform", "cube_90")

if TYPE_CHECKING:
  import viser


class InHandReorientCommand(CommandTerm):
  cfg: InHandReorientCommandCfg

  def __init__(self, cfg: InHandReorientCommandCfg, env):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    palm_ids, _ = self.robot.find_bodies(cfg.palm_body_pattern)
    if not palm_ids:
      raise ValueError(
        f"InHandReorientCommand: no palm body matched pattern "
        f"'{cfg.palm_body_pattern}' on entity '{cfg.robot_entity_name}'."
      )
    self.palm_body_id = int(palm_ids[0])

    tag_ids, _ = self.robot.find_sites(cfg.tag_site_pattern)
    if not tag_ids:
      raise ValueError(
        f"InHandReorientCommand: no tag site matched '{cfg.tag_site_pattern}' "
        f"on entity '{cfg.robot_entity_name}'."
      )
    self.tag_site_id = int(tag_ids[0])

    self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal_quat_w[:, 0] = 1.0

    if cfg.goal_mode not in _GOAL_MODES:
      raise ValueError(
        f"InHandReorientCommand: goal_mode must be one of {_GOAL_MODES}, "
        f"got '{cfg.goal_mode}'."
      )
    # cube_90 mode: goals are one of the 24 cube rotations (multiples of 90 deg)
    # in tag frame. Each new goal is a neighbour of the previous goal whose
    # rotation distance lies in (0, goal_max_angle].
    self._cube_quats = cube_rotation_quats(self.device)
    pair_angle = quat_error_magnitude(
      self._cube_quats.unsqueeze(1).expand(-1, 24, -1).reshape(-1, 4),
      self._cube_quats.unsqueeze(0).expand(24, -1, -1).reshape(-1, 4),
    ).reshape(24, 24)
    self._cube_neighbor_mask = (pair_angle > 1e-3) & (
      pair_angle <= cfg.goal_max_angle + 1e-3
    )
    if cfg.goal_axes is not None:
      # Only +-90 deg turns about the listed tag-frame axes: goal_rel_new =
      # R_axis(+-90) * goal_rel_old (left-multiplied => rotation in tag frame).
      half = math.sqrt(0.5)
      mask = torch.zeros(24, 24, dtype=torch.bool, device=self.device)
      axis_table = torch.full((24, 24), -1, dtype=torch.long, device=self.device)
      for axis in cfg.goal_axes:
        if axis not in (0, 1, 2):
          raise ValueError(f"goal_axes entries must be 0, 1 or 2, got {axis}.")
        for sign in (1.0, -1.0):
          turn = torch.zeros(4, device=self.device)
          turn[0] = half
          turn[1 + axis] = sign * half
          rotated = quat_mul(turn.expand(24, -1), self._cube_quats)
          dst = nearest_cube_rotation_index(rotated, self._cube_quats)
          mask[torch.arange(24), dst] = True
          axis_table[torch.arange(24), dst] = axis
      self._cube_neighbor_mask = mask
      self._cube_neighbor_axis = axis_table
    else:
      self._cube_neighbor_axis = torch.full(
        (24, 24), -1, dtype=torch.long, device=self.device
      )
    if cfg.goal_mode == "cube_90" and not self._cube_neighbor_mask[0].any():
      raise ValueError(
        "InHandReorientCommand: goal_max_angle must be >= pi/2 in cube_90 mode."
      )
    self.goal_cube_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    # Per-axis bookkeeping (cube_90 + goal_axes only): which tag axis the
    # current goal turns about (-1 = unknown), and per-episode goal / success
    # counts per axis, logged as Metrics/reorient_command/{goals,reach}_<axis>.
    self.goal_axis = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.axis_goal_count = torch.zeros(self.num_envs, 3, device=self.device)
    self.axis_reach_count = torch.zeros(self.num_envs, 3, device=self.device)

    self.hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self.goal_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    self.goal_reach_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self.in_success_window = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self._reward_window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self.reward_hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._reward_hold_counter_snapshot = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self._success_achieved = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._goal_switched = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._within_threshold = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["goal_reach_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ori_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hold_counter"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_timer"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_success_window"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["window_timer"] = torch.zeros(self.num_envs, device=self.device)
    self._axis_metric_names = (
      cfg.goal_axis_names if cfg.goal_axes is not None else ()
    )
    for name in self._axis_metric_names:
      self.metrics[f"goals_{name}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"reach_{name}"] = torch.zeros(self.num_envs, device=self.device)

    self._visualization = ReorientCommandVisualization(entity_name=self.cfg.entity_name)

  @property
  def command(self) -> torch.Tensor:
    return self.goal_quat_w

  @property
  def goal_quat(self) -> torch.Tensor:
    return self.goal_quat_w

  def _update_metrics(self) -> None:
    self.metrics["goal_reach_count"] = self.goal_reach_count.float()
    ori_err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    self.metrics["ori_error"] = ori_err
    self.metrics["hold_counter"] = self.hold_counter.float()
    self.metrics["goal_timer"] = self.goal_timer.float()
    self.metrics["in_success_window"] = self.in_success_window.float()
    self.metrics["window_timer"] = self.window_timer.float()
    for i, name in enumerate(self._axis_metric_names):
      self.metrics[f"goals_{name}"] = self.axis_goal_count[:, i].clone()
      self.metrics[f"reach_{name}"] = self.axis_reach_count[:, i].clone()
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    """Sample new goal_quat_w for the given envs.

    Goal is drawn uniformly on SO(3) in tag frame to match deploy semantics
    (the deploy side samples goal_quat directly in tag frame without any
    wrist transform).
    """
    n = len(env_ids)
    tag_quat_w = self.robot.data.site_pose_w[env_ids, self.tag_site_id, 3:7]
    if self.cfg.goal_mode == "cube_90":
      tag_rel_goal = self._sample_cube_90_goal(env_ids)
    else:
      tag_rel_goal = random_quat_uniform(n, self.device)
    self.goal_quat_w[env_ids] = quat_mul(tag_quat_w, tag_rel_goal)

  def _sample_cube_90_goal(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Pick a 90-deg-multiple neighbour of the current reference orientation."""
    ref_idx = self.goal_cube_idx[env_ids]
    weights = self._cube_neighbor_mask[ref_idx].float()
    new_idx = torch.multinomial(weights, 1).squeeze(-1)
    self.goal_cube_idx[env_ids] = new_idx
    axis = self._cube_neighbor_axis[ref_idx, new_idx]
    self.goal_axis[env_ids] = axis
    known = axis >= 0
    if known.any():
      self.axis_goal_count[env_ids[known], axis[known]] += 1.0
    return self._cube_quats[new_idx]

  def _snap_reference_to_cube(self, env_ids: torch.Tensor) -> None:
    """Use the cube's current orientation (snapped to 90 deg) as the reference."""
    tag_quat_w = self.robot.data.site_pose_w[env_ids, self.tag_site_id, 3:7]
    # Read the free-joint qpos, not root_link_quat_w: on reset this runs right
    # after the reset event wrote qpos but before mj_forward refreshes xquat.
    quat_adr = self.object.indexing.free_joint_q_adr[3:7]
    cube_quat_w = self.object.data.data.qpos[:, quat_adr][env_ids]
    cube_in_tag = quat_mul(quat_inv(tag_quat_w), cube_quat_w)
    self.goal_cube_idx[env_ids] = nearest_cube_rotation_index(
      cube_in_tag, self._cube_quats
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.axis_goal_count[env_ids] = 0.0
    self.axis_reach_count[env_ids] = 0.0
    if self.cfg.goal_mode == "cube_90":
      self._snap_reference_to_cube(env_ids)
    self._sample_goal_in_world(env_ids)
    self.hold_counter[env_ids] = 0
    self.goal_timer[env_ids] = 0
    self.goal_reach_count[env_ids] = 0
    self.in_success_window[env_ids] = False
    self.window_timer[env_ids] = 0
    self._reward_window_timer[env_ids] = 0
    self.reward_hold_counter[env_ids] = 0
    self._reward_hold_counter_snapshot[env_ids] = 0
    self._success_achieved[env_ids] = False
    self._goal_switched[env_ids] = False
    self._within_threshold[env_ids] = False

  def _min_goal_steps(self) -> int:
    return max(int(self.cfg.min_goal_interval / self._env.step_dt), 0)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    selected = slice(None) if env_ids is None else env_ids
    goal_timer = self.goal_timer[selected] + 1
    ori_err = quat_error_magnitude(
      self.object.data.root_link_quat_w[selected], self.goal_quat_w[selected]
    )
    within_threshold = ori_err < self.cfg.success_threshold

    approaching_mask = ~self.in_success_window[selected]
    hold_counter = self.hold_counter[selected]
    hold_counter = torch.where(
      approaching_mask & within_threshold,
      hold_counter + 1,
      torch.where(approaching_mask, torch.zeros_like(hold_counter), hold_counter),
    )

    just_succeeded = approaching_mask & (hold_counter >= self.cfg.success_hold_steps)

    in_success_window = self.in_success_window[selected] | just_succeeded
    window_timer = self.window_timer[selected]
    window_timer = torch.where(
      just_succeeded, torch.zeros_like(window_timer), window_timer
    )
    hold_counter = torch.where(
      just_succeeded, torch.zeros_like(hold_counter), hold_counter
    )
    self.goal_reach_count[selected] += just_succeeded.long()
    succ_ids = just_succeeded.nonzero(as_tuple=False).squeeze(-1)
    if env_ids is not None:
      succ_ids = env_ids[succ_ids]
    if succ_ids.numel() > 0:
      succ_axis = self.goal_axis[succ_ids]
      known = succ_axis >= 0
      self.axis_reach_count[succ_ids[known], succ_axis[known]] += 1.0

    self._success_achieved[selected] = just_succeeded

    window_mask = in_success_window
    window_timer = torch.where(window_mask, window_timer + 1, window_timer)

    reward_hold_counter = self.reward_hold_counter[selected]
    reward_hold_counter = torch.where(
      window_mask & within_threshold,
      reward_hold_counter + 1,
      reward_hold_counter,
    )
    reward_hold_counter = torch.where(
      just_succeeded, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    min_goal_steps = self._min_goal_steps()
    should_switch = (
      window_mask
      & (window_timer >= self.cfg.goal_switch_delay)
      & (goal_timer >= min_goal_steps)
    )

    advance_ids = should_switch.nonzero(as_tuple=False).squeeze(-1)
    if env_ids is not None:
      advance_ids = env_ids[advance_ids]
    if advance_ids.numel() > 0:
      self._sample_goal_in_world(advance_ids)

    self._goal_switched[selected] = should_switch

    self._reward_window_timer[selected] = window_timer
    self._reward_hold_counter_snapshot[selected] = reward_hold_counter

    in_success_window = torch.where(
      should_switch, torch.zeros_like(in_success_window), in_success_window
    )
    window_timer = torch.where(
      should_switch, torch.zeros_like(window_timer), window_timer
    )
    hold_counter = torch.where(
      should_switch, torch.zeros_like(hold_counter), hold_counter
    )
    goal_timer = torch.where(should_switch, torch.zeros_like(goal_timer), goal_timer)
    reward_hold_counter = torch.where(
      should_switch, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    self._within_threshold[selected] = within_threshold

    self.hold_counter[selected] = hold_counter
    self.goal_timer[selected] = goal_timer
    self.in_success_window[selected] = in_success_window
    self.window_timer[selected] = window_timer
    self.reward_hold_counter[selected] = reward_hold_counter

  def _debug_vis_impl(self, visualizer) -> None:
    self._visualization.draw_debug_visuals(
      visualizer=visualizer,
      num_envs=self.num_envs,
      palm_pose_w=self.robot.data.body_link_pose_w[:, self.palm_body_id, :],
      tag_pose_w=self.robot.data.site_pose_w[:, self.tag_site_id, :],
      object_pos_w=self.object.data.root_link_pos_w,
      goal_quat_w=self.goal_quat_w,
      policy_status_for_env=self._policy_status_for_env,
    )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx,
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    # This panel only displays status; mjlab's editing callbacks are unused.
    self._visualization.create_gui(
      name=name,
      server=server,
      get_env_idx=get_env_idx,
    )
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _policy_status_for_env(self, env_idx: int) -> tuple[float, bool]:
    ori_error_rad = float(
      quat_error_magnitude(
        self.object.data.root_link_quat_w[env_idx : env_idx + 1],
        self.goal_quat_w[env_idx : env_idx + 1],
      )[0].item()
    )
    is_success = ori_error_rad < self.cfg.success_threshold
    return ori_error_rad, is_success

  @property
  def within_threshold(self) -> torch.Tensor:
    return self._within_threshold

  @property
  def success_achieved(self) -> torch.Tensor:
    return self._success_achieved

  @property
  def goal_switched(self) -> torch.Tensor:
    return self._goal_switched

  @property
  def reward_window_timer(self) -> torch.Tensor:
    return self._reward_window_timer

  @property
  def reward_hold_counter_snapshot(self) -> torch.Tensor:
    return self._reward_hold_counter_snapshot


@dataclass(kw_only=True)
class InHandReorientCommandCfg(CommandTermCfg):
  entity_name: str
  success_threshold: float = 0.2
  success_hold_steps: int = 5
  goal_switch_delay: int = 20
  min_goal_interval: float = 0.0
  debug_vis: bool = False
  robot_entity_name: str = "robot"
  palm_body_pattern: str = ".*_palm_link"
  tag_site_pattern: str = ".*_wrist_tag"
  goal_mode: str = "uniform"
  """"uniform": goals uniform on SO(3) in tag frame (default).
  "cube_90": goals are multiples of 90 deg in tag frame (24 cube rotations)."""
  goal_max_angle: float = math.pi / 2
  """cube_90 only: max rotation (rad) between consecutive goals. pi/2 gives the
  6 single quarter-turns; 2*pi/3 adds diagonal turns; pi allows any of the 23."""
  goal_axes: tuple[int, ...] | None = None
  """cube_90 only: if set, each new goal is a +-90 deg turn about one of these
  tag-frame axes (0=x, 1=y, 2=z) and goal_max_angle is ignored."""
  goal_axis_names: tuple[str, str, str] = ("x", "y", "z")
  """Names of the tag axes 0/1/2 used in the per-axis goal/reach metrics."""

  def build(self, env) -> InHandReorientCommand:
    return InHandReorientCommand(self, env)
