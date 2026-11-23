"""MyHand_Reorient: thumb action range reaches the cube (per-joint scale/center)."""

import pytest

pytest.importorskip("mujoco_warp")

from wuji_mjlab.tasks.reorient.config.my_hand import env_cfgs as ec


def _resolve(table: dict[str, float], name: str) -> float | None:
  import re

  for pattern, value in table.items():
    if re.fullmatch(pattern, name):
      return value
  return None


def test_thumb_targets_cover_cube_contact_pose():
  # Kinematic sweep: thumb tip on the cube needs CMC ~1.8-2.4 rad.
  name = "right_finger1_joint1"
  center = _resolve(ec.MY_HAND_ACTION_CENTER, name)
  scale = _resolve(ec.MY_HAND_ACTION_SCALE, name)
  assert center + scale >= 2.4
  assert center - scale <= ec.MY_HAND_INIT_JOINT_POS[".*_finger1_joint1"]


def test_fingers_keep_default_range():
  name = "right_finger2_joint1"
  assert _resolve(ec.MY_HAND_ACTION_CENTER, name) is None
  assert _resolve(ec.MY_HAND_ACTION_SCALE, name) == 0.5


def test_env_cfg_wires_thumb_terms():
  cfg = ec.my_hand_reorient_env_cfg(num_envs=4)
  act = cfg.actions["joint_pos"]
  assert act.action_scale is ec.MY_HAND_ACTION_SCALE
  assert act.action_center is ec.MY_HAND_ACTION_CENTER
  assert "thumb_reach" in cfg.rewards
  assert cfg.rewards["hand_pose"].params["asset_cfg"].joint_names == (
    ".*_finger[2-5]_joint.*",
  )

