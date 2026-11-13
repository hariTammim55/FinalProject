"""MyHand_Reorient: in-hand cube reorientation with the custom RUKA-design hand.

The robot is assets/robots/myHandRuka: 15 joints, 11 servo-derived actuators
(each finger: MCP + one tendon pulling PIP and DIP together, DIP coupled to
PIP; thumb: CMC, MCP, IP). No touch sensors in the observations.

History: this replaces WujiMyHand_Reorient (15 independent joint actuators,
removed) and was registered as WujiMyHandRuka_Reorient before. Checkpoints of
the 15-actuator task (logs/*_MyHand_Reorient from 2026-11-12) do not load
here (15 vs 11 actions).
"""

import math
from collections.abc import Callable
from functools import partial

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from wuji_mjlab.assets.objects.merah_putih_cube.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.myHandRuka.my_hand_ruka_cfg import (
  get_my_hand_ruka_cfg,
)
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

# Axis curriculum (see my_hand_reorient_env_cfg). Set INITIAL_UNLOCKED to 2
# when resuming a run that already passed the yaw stage: the curriculum state
# is not saved in checkpoints and otherwise restarts at yaw only.
# Tag axes whose +-90 deg turns are used as goals (training and play).
# Current stage: yaw and pitch only, both directions. Add
# MY_HAND_TAG_ROLL_AXIS to bring roll back.
MY_HAND_GOAL_AXES = (MY_HAND_TAG_YAW_AXIS, MY_HAND_TAG_PITCH_AXIS)
# Unlock order; axes not in MY_HAND_GOAL_AXES are skipped.
MY_HAND_AXIS_UNLOCK_ORDER = (
  MY_HAND_TAG_YAW_AXIS,
  MY_HAND_TAG_PITCH_AXIS,
  MY_HAND_TAG_ROLL_AXIS,
)
# 2 = yaw and pitch active from the start (resuming from model_4999.pt, which
# already does yaw+-90 and pitch-90). Use 1 for a fresh run.
MY_HAND_AXIS_INITIAL_UNLOCKED = 2
# An axis counts as mastered once ONE of its directions succeeds this often
# (the other direction = 3 turns this way, see cube_turn_planner).
MY_HAND_AXIS_UNLOCK_THRESHOLD = 0.5
# Give-up: a direction still below GIVE_UP_RATE success GIVE_UP_ITERS after
# its axis unlocked is sampled only at the weight floor. Disabled (rate 0.0)
# for this stage so pitch+90 keeps being practised the most; set 0.05 to let
# the curriculum drop directions the hand cannot do.
MY_HAND_AXIS_GIVE_UP_ITERS = 1500
MY_HAND_AXIS_GIVE_UP_RATE = 0.0
# Unreached goals are abandoned after this long (first run: envs stuck on an
# impossible pitch+90 goal spent ~800 of 1000 steps on it). A successful
# quarter-turn should take well under this.
MY_HAND_GOAL_TIMEOUT_S = 15.0
_STEPS_PER_ITER = 40  # rsl_rl num_steps_per_env

# Fingers 2..5: the tip joint (joint3) is tied 1:1 to joint2, so both start
# at the same angle and get no independent reset noise (a mismatch would be
# snapped together by the coupling at reset). Thumb unchanged.
MY_HAND_INIT_JOINT_POS = {
  ".*_finger1_joint1": 0.40,
  ".*_finger1_joint2": 0.30,
  ".*_finger1_joint3": 0.50,
  ".*_finger[2-5]_joint1": 0.60,
  ".*_finger[2-5]_joint[2-3]": 0.45,
}
MY_HAND_JOINT_RESET_RANGES = {
  ".*_joint1": (-0.3, -0.1),
  ".*_joint2": (0.0, 0.0),
  ".*_finger1_joint3": (-0.3, -0.1),
  ".*_finger[2-5]_joint3": (0.0, 0.0),
}
# Torque penalty is relative to each servo's limit (actuator forcerange).
# Random exploration (std ~1) gives sum (tau/limit)^2 ~0.5 per step, i.e.
# ~-13/s at Wuji's -24, more than all task rewards combined; -6 keeps it a
# mild "don't saturate the servos" term (monitor Episode_Reward/torque).
MY_HAND_TORQUE_WEIGHT = -6.0
# Thumb on the cube. Without it the policy (model_5998 of the first run)
# rotated the cube with the 4 strong coupled fingers + palm only: thumb in
# contact 7% of the time vs 54% for the old 15-actuator hand, thumb joints nearly
# static at their default pose (not torque-limited, 0% saturation). Only the
# thumb's distal segment counts (Thumb_IP_Link = right_finger1_link4, the
# thumb counterpart of the fingers' tip *_PIP_Link); contacts with the CMC
# and MCP links are not rewarded. model_5998 touches with it only 0.6% of the
# time (old 15-actuator hand: 9%), so raise the weight if
# Episode_Reward/thumb_contact stays near 0.
# Same scale as tip_contact (1.0 per second at full contact).
MY_HAND_THUMB_CONTACT_WEIGHT = 1.0


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
  robot_cfg = get_my_hand_ruka_cfg()
  robot_cfg.spec_fn = partial(_add_wrist_tag, robot_cfg.spec_fn)
  return robot_cfg


def my_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 4096
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
    joint_reset_position_ranges=MY_HAND_JOINT_RESET_RANGES,
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
  # Each new goal is a +-90 deg turn about one of MY_HAND_GOAL_AXES (currently
  # yaw and pitch; any orientation, roll included, is reachable from those at
  # deploy time via cube_turn_planner.plan_cube_turns).
  cmd.goal_axes = MY_HAND_GOAL_AXES
  cmd.goal_axis_names = ("roll", "yaw", "pitch")
  # Staged curriculum inside this one task: axes unlock in
  # MY_HAND_AXIS_UNLOCK_ORDER once each active axis has a direction above
  # MY_HAND_AXIS_UNLOCK_THRESHOLD. Active turns are sampled more often the
  # more they fail, so the weaker direction (pitch+90) gets the most practice.
  # Progress: Metrics/reorient_command/curr_active_axes and curr_rate_<axis>_<sign>.
  if not play:
    cmd.axis_curriculum = True
    cmd.axis_unlock_order = tuple(
      axis for axis in MY_HAND_AXIS_UNLOCK_ORDER if axis in MY_HAND_GOAL_AXES
    )
    cmd.axis_initial_unlocked = MY_HAND_AXIS_INITIAL_UNLOCKED
    cmd.axis_unlock_threshold = MY_HAND_AXIS_UNLOCK_THRESHOLD
    cmd.axis_give_up_steps = MY_HAND_AXIS_GIVE_UP_ITERS * _STEPS_PER_ITER
    cmd.axis_give_up_rate = MY_HAND_AXIS_GIVE_UP_RATE
  cmd.goal_timeout_s = MY_HAND_GOAL_TIMEOUT_S
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
  # Without abduction the palm only supports the cube; rotating it needs
  # fingertips pushing on it. Rewards up to 2 tips in contact (not a clamp).
  cfg.rewards["tip_contact"] = RewardTermCfg(
    func=mdp.TipContactReward,
    weight=1.0,
    params={
      "robot_cfg": SceneEntityCfg("robot", site_names=(".*_finger[1-5]_tip",)),
      "sensor_cfg": SceneEntityCfg("tip_object_contact"),
      "target_tips": 2,
    },
  )
  # Thumb tip on the cube (see MY_HAND_THUMB_CONTACT_WEIGHT).
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    ContactSensorCfg(
      name="thumb_object_contact",
      primary=ContactMatch(
        mode="body", pattern=".*_finger1_link4", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
      fields=("found",),
      reduce="netforce",
      num_slots=1,
    ),
  )
  cfg.rewards["thumb_contact"] = RewardTermCfg(
    func=mdp.sensor_contact_found,
    weight=MY_HAND_THUMB_CONTACT_WEIGHT,
    params={"sensor_name": "thumb_object_contact"},
  )
  cfg.rewards["torque"].params["normalize_by_forcerange"] = True
  cfg.rewards["torque"].weight = MY_HAND_TORQUE_WEIGHT
  cfg.scene.entities = {
    "robot": get_my_hand_rig_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = EntityCfg.InitialStateCfg(
    pos=MY_HAND_ROOT_POS,
    rot=MY_HAND_ROOT_ROT,
    joint_pos=MY_HAND_INIT_JOINT_POS,
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