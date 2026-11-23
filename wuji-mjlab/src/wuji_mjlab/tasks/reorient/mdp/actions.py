# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


def warmup_steps(warmup_time_s: float, step_dt: float) -> int:
  """Control steps to hold at the default pose, as an exact integer count.

  The bound must not be tested by comparing ``step_index * step_dt`` against
  ``warmup_time_s`` in floating point.
  Mirrored in ``deploy/reorient/wuji_reorient_deploy/obs_builder.py``; a test pins them equal.
  """
  if step_dt <= 0.0:
    raise ValueError(f"step_dt must be positive, got {step_dt}")
  return max(0, math.ceil(warmup_time_s / step_dt - 1e-9))


class JointPositionOffsetEMAAction(JointPositionAction):
  cfg: "JointPositionOffsetEMAActionCfg"

  def __init__(self, cfg: "JointPositionOffsetEMAActionCfg", env):
    super().__init__(cfg, env)

    self._ema_alpha = cfg.ema_alpha
    self._warmup_time_s = cfg.warmup_time_s
    self._warmup_steps = warmup_steps(cfg.warmup_time_s, env.step_dt)

    self._default_joint_pos = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()
    joint_names = [self._entity.joint_names[int(i)] for i in self._target_ids]
    self._action_scale = self._per_joint(cfg.action_scale, joint_names, None)
    # Action 0 maps to the center; the reset/warmup pose stays the default.
    self._action_center = self._per_joint(
      cfg.action_center, joint_names, self._default_joint_pos
    )

    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft_limits[..., 0]
    self._upper_limits = soft_limits[..., 1]

    self._prev_target = self._default_joint_pos.clone()
    self._processed_actions.copy_(self._default_joint_pos)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)

    raw_target = self._action_center + clamped * self._action_scale
    raw_target = torch.clamp(raw_target, self._lower_limits, self._upper_limits)

    smoothed = (
      self._ema_alpha * raw_target + (1.0 - self._ema_alpha) * self._prev_target
    )

    in_warmup = (self._env.episode_length_buf < self._warmup_steps).unsqueeze(-1)
    self._processed_actions = torch.where(in_warmup, self._default_joint_pos, smoothed)
    self._prev_target = self._processed_actions.clone()

  def _per_joint(self, value, joint_names, base):
    """Resolve a float or ``{regex: value}`` (first match wins) per joint.

    ``None`` returns ``base``. Joints no pattern matches keep ``base``; with no
    ``base`` (action_scale) every joint must be matched.
    """
    if value is None:
      return base
    if not isinstance(value, dict):
      return value
    out = (
      base.clone()
      if base is not None
      else torch.full_like(self._default_joint_pos, float("nan"))
    )
    for j, name in enumerate(joint_names):
      for pattern, v in value.items():
        if re.fullmatch(pattern, name):
          out[:, j] = v
          break
    if torch.isnan(out).any():
      missing = [n for j, n in enumerate(joint_names) if torch.isnan(out[0, j])]
      raise ValueError(f"No action_scale pattern matches joints {missing}")
    return out

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._prev_target[env_ids] = self._default_joint_pos[env_ids]
    self._processed_actions[env_ids] = self._default_joint_pos[env_ids]


@dataclass(kw_only=True)
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
  """Configuration for offset + EMA action with warmup."""

  # Float, or {joint-name regex: scale} (first match wins, all joints matched).
  action_scale: float | dict[str, float] = 0.5
  # Joint target for action 0, {regex: rad}; unmatched joints (and None) use
  # the default joint pos. Lets a joint reach past default +- action_scale
  # without moving its reset pose.
  action_center: dict[str, float] | None = None
  ema_alpha: float = 0.5
  warmup_time_s: float = 0.4

  def build(self, env) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)
