"""Reorient environment configuration for the custom hand MJCF."""

import math
from collections.abc import Callable
from functools import partial

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from wuji_mjlab.assets.objects.merah_putih_cube.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.myHand.my_hand_cfg import get_my_hand_cfg
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_WRIST_TAG_POS,
  REORIENT_WRIST_TAG_ROT,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg

MY_HAND_ROOT_POS = (0.0, 0.0, 0.5)

# Tilt the whole hand about world Y so gravity pulls the cube toward the thumb.
# Flat (0 deg), the hand geometry makes the cube drift ~60 mm away from the
# thumb within 20 s even with no finger motion, and the fingers (no abduction)
# cannot pull it back. CPU probe with random finger motion, 20 s per trial:
#   tilt   0: 54% drop | -5: 1% drop | -8: 12% | -12: 77% | -15: 100%
MY_HAND_TILT_TOWARD_THUMB_DEG = 3.0
_tilt = math.radians(-MY_HAND_TILT_TOWARD_THUMB_DEG)
MY_HAND_ROOT_ROT = (math.cos(_tilt / 2), 0.0, math.sin(_tilt / 2), 0.0)

# Cube spawn point, given relative to the hand root before tilting, then
# rotated with the hand so it stays above the same spot on the fingers.
_CUBE_OFFSET_FROM_ROOT = (0.0, -0.04, 0.08)
MY_HAND_CUBE_INIT_POS = (
  MY_HAND_ROOT_POS[0] + math.cos(_tilt) * _CUBE_OFFSET_FROM_ROOT[0]
  + math.sin(_tilt) * _CUBE_OFFSET_FROM_ROOT[2],
  MY_HAND_ROOT_POS[1] + _CUBE_OFFSET_FROM_ROOT[1],
  MY_HAND_ROOT_POS[2] - math.sin(_tilt) * _CUBE_OFFSET_FROM_ROOT[0]
  + math.cos(_tilt) * _CUBE_OFFSET_FROM_ROOT[2],
)

# Palm-frame axis that points out of the palm surface (toward the cube).
# Wuji Hand uses palm +X (REORIENT_PALM_NORMAL_AXIS = 0), but in this MJCF the
# Palm_Link is rotated by euler(-80.7, 0, 180), so its +Y axis points up
# (world ≈ (0, -0.16, 0.99)). The cage and cube-height metric depend on this.
MY_HAND_PALM_NORMAL_AXIS = 1

# Tag-frame axes (tag = palm * REORIENT_WRIST_TAG_ROT): x ~ along the fingers
# (roll), y = palm normal (yaw), z = lateral (pitch).
MY_HAND_TAG_ROLL_AXIS = 0
MY_HAND_TAG_YAW_AXIS = 1
MY_HAND_TAG_PITCH_AXIS = 2
MY_HAND_SUCCESS_THRESHOLD = 0.4  # rad (~23 deg)


def _add_wrist_tag(base_spec_fn: Callable[[], mujoco.MjSpec]) -> mujoco.MjSpec:
  spec = base_spec_fn()
  spec.body("right_palm_link").add_site(
    name="right_wrist_tag",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=REORIENT_WRIST_TAG_POS,
    quat=REORIENT_WRIST_TAG_ROT,
    size=(0.0252, 0.0252, 0.0006),
    rgba=(1.0, 0.9, 0.0, 0.5),
    group=1,
  )
  return spec


def get_my_hand_rig_cfg() -> EntityCfg:
  robot_cfg = get_my_hand_cfg()
  robot_cfg.spec_fn = partial(_add_wrist_tag, robot_cfg.spec_fn)
  return robot_cfg


def my_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    soft_pad_geoms=(),
    geom_size_dr_geoms=None,
    undesired_object_contact_bodies=(
      ".*_palm_link",
      ".*_finger[1-5]_link[2-3]",
    ),
    joint_reset_position_ranges={
      ".*_joint1": (-0.3, -0.1),
      ".*_joint2": (0.0, 0.0),
      ".*_joint3": (-0.3, -0.1),
    },
    cage_up_axis=MY_HAND_PALM_NORMAL_AXIS,
  )
  # Keep the robot-pose reset event: for a fixed-base (mocap) robot it is the
  # ONLY thing that moves each hand to its env origin. Removing it stacks every
  # hand at (0, 0, 0.5) while each cube spawns at its own env origin, so in
  # almost every env the cube is ~0.5 m away and cage_drop ends the episode
  # after ~10 steps. Wuji's pitch tilt is tuned for its own mount, so disable it.
  cfg.events["reset_robot_pose"].params["pose_range"] = {}

  # Goals are multiples of 90 deg (the 24 face-aligned cube orientations, in
  # tag frame); each new goal is a single quarter-turn from the previous one.
  # Raise goal_max_angle to 2*pi/3 or pi once single quarter-turns succeed.
  cmd = cfg.commands["reorient_command"]
  cmd.goal_mode = "cube_90"
  cmd.goal_max_angle = math.pi / 2
  # All three axes: each new goal is a +-90 deg turn about roll, yaw or pitch
  # (6 quarter-turns, equally likely). Yaw is by far the easiest for this
  # flexion-only hand (random-action probe: pitch/roll never exceeded ~50 deg),
  # so expect reach_yaw to rise first. Use (MY_HAND_TAG_YAW_AXIS,) for the
  # yaw-only stage.
  cmd.goal_axes = (
    MY_HAND_TAG_ROLL_AXIS,
    MY_HAND_TAG_YAW_AXIS,
    MY_HAND_TAG_PITCH_AXIS,
  )
  cmd.goal_axis_names = ("roll", "yaw", "pitch")
  # Looser success window for the first successes (Wuji uses 0.2 rad).
  cmd.success_threshold = MY_HAND_SUCCESS_THRESHOLD
  # Start the cube face-aligned too (with ~5 deg noise), otherwise a randomly
  # dropped cube settles at an arbitrary yaw and "90 deg" goals are not 90 deg.
  cfg.events["reset_object_pose"].params["orientation_mode"] = "cube_90"
  cfg.events["reset_object_pose"].params["rot_noise"] = math.radians(5.0)

  # Reward progress toward the goal instead of mostly rewarding the absolute
  # error: with only orientation_alignment, holding the cube still 90 deg from
  # the goal already pays well, so the policy learned to never rotate it.
  # Episode sum of orientation_progress = weight * (rad of error removed).
  cfg.rewards["orientation_alignment"].weight = 5.0
  cfg.rewards["orientation_progress"] = RewardTermCfg(
    func=mdp.OrientationProgressReward,
    weight=20.0,
    params={"command_name": "reorient_command", "max_rate": 4.0},
  )
  # Exploration noise is penalized as "non-smooth" motion; Wuji's -1.0 made
  # action_rate the largest negative term for this hand and discouraged moving.
  cfg.rewards["action_rate"].weight = -0.3
  cfg.scene.entities = {
    "robot": get_my_hand_rig_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = EntityCfg.InitialStateCfg(
    pos=MY_HAND_ROOT_POS,
    rot=MY_HAND_ROOT_ROT,
    joint_pos={
      ".*_finger1_joint1": 0.40,
      ".*_finger1_joint2": 0.30,
      ".*_finger1_joint3": 0.50,
      ".*_finger[2-5]_joint1": 0.60,
      ".*_finger[2-5]_joint2": 0.40,
      ".*_finger[2-5]_joint3": 0.50,
    },
    joint_vel={".*": 0.0},
  )
  cfg.scene.entities["object"].init_state = EntityCfg.InitialStateCfg(
    pos=MY_HAND_CUBE_INIT_POS,
    rot=(1.0, 0.0, 0.0, 0.0),
  )
  if play:
    cfg.scene.num_envs = 1
  cfg.viewer.body_name = "right_palm_link"
  return cfg