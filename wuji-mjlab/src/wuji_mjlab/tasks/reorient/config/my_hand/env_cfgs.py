"""Reorient environment configuration for the custom hand MJCF."""

from collections.abc import Callable
from functools import partial

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.myHand.my_hand_cfg import get_my_hand_cfg
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_PALM_NORMAL_AXIS,
  REORIENT_WRIST_TAG_POS,
  REORIENT_WRIST_TAG_ROT,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg

MY_HAND_ROOT_POS = (0.0, 0.0, 0.5)
MY_HAND_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)
MY_HAND_CUBE_INIT_POS = (-0.00, -0.04, 0.58)


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
    cage_up_axis=REORIENT_PALM_NORMAL_AXIS,
  )
  cfg.events.pop("reset_robot_pose", None)
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