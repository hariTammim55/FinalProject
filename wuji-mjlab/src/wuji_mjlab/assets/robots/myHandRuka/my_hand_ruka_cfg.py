"""MJLab adapter for the custom hand with RUKA tendon actuation (11 servos).

Same geometry as ``assets/robots/myHand`` (15 independent joint actuators),
but actuated like the real RUKA-design hand: each finger has one servo on MCP
and ONE tendon pulling PIP and DIP together (DIP follows PIP through a joint
equality), the thumb has one servo per joint. Gains and torque limits in
``mjcf/hand_assembly_ruka.xml`` come from the Dynamixel servos, see
``ruka_actuator_params``.

Joint / body names are converted to the task convention by the myHand
adapter, so the task sees right_finger<f>_joint<1..3> as before; the policy
just has 11 actions instead of 15 (no action for right_finger[2-5]_joint3).
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

from wuji_mjlab.assets.robots.myHand.my_hand_cfg import (
  _apply_task_names,
  _get_assets,
)

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


_XML_PATH = Path(__file__).resolve().parent / "mjcf" / "hand_assembly_ruka.xml"

# Dynamixel stall torque at 12 V (ROBOTIS e-manual) and the tendon spool.
SERVO_STALL_TORQUE_NM = {"XM430-W210-T": 3.0, "XM430-W350-T": 4.1}
THUMB_SERVO = "XM430-W350-T"
FINGER_SERVO = "XM430-W210-T"
PULLEY_RADIUS_M = 0.0255 / 2
# Fraction of stall torque the servo may use. 1.0 = no current limit (the
# current plan). With a Current Limit register value L (1 unit = 2.69 mA,
# stall current ~2.3 A at 12 V), use L * 2.69e-3 / 2.3.
SERVO_TORQUE_FRACTION = 1.0
# Dynamixel Position P Gain (factory default 800). The sim stiffness is
# derived from it, so keep the real servos at this gain for sim2real:
#   motor torque ~= (KPP / 128) * error[pulse] / 885 * stall_torque
SERVO_POSITION_P_GAIN = 800.0
# kv / kp, same ratio as the 15-actuator MJCF (kp 0.3735, kv 0.0188).
KV_OVER_KP = 0.0504

# Tendon moment arm at each driven joint (m), estimated from RUKA's
# calibrated motor ranges (RUKA/ruka_hand/utils/constants.py,
# MOTOR_RANGES_RIGHT) with this spool: tendon travel / joint range, averaged
# over the 4 fingers. "PIP+DIP" is the sum over both joints (one tendon).
# Accuracy ~+-30%; replace with CAD values when available.
TENDON_MOMENT_ARM_M = {
  "finger_MCP": 0.0084,
  "finger_PIP+DIP": 0.0124,
  "thumb_CMC": 0.0053,
  "thumb_MCP": 0.0075,
  "thumb_IP": 0.0053,
}

# Task joint names: right_finger<f>_joint<1..3>, f = 1 thumb, 2..5 fingers.
_THUMB_ARM_KEYS = {
  "joint1": "thumb_CMC",
  "joint2": "thumb_MCP",
  "joint3": "thumb_IP",
}
_FINGER_ARM_KEYS = {"joint1": "finger_MCP", "joint2": "finger_PIP+DIP"}


def ruka_actuator_params(joint_name: str) -> tuple[float, float, float]:
  """(kp [N·m/rad], kv [N·m·s/rad], torque limit [N·m]) for a driven joint.

  ``joint_name`` is a task joint name (right_finger<f>_joint<n>). The servo
  sees the joint through the tendon: joint torque = tendon force * arm,
  tendon force = motor torque / spool radius, and a joint motion dq turns the
  spool by dq * arm / radius, so the joint-side stiffness is the servo
  stiffness times (arm / radius)^2. The values written in the MJCF must
  match these (checked by tests/test_my_hand_ruka_actuation.py).
  """
  finger, joint = joint_name.split("_")[-2:]
  is_thumb = finger == "finger1"
  key = (_THUMB_ARM_KEYS if is_thumb else _FINGER_ARM_KEYS)[joint]
  arm = TENDON_MOMENT_ARM_M[key]
  stall = SERVO_STALL_TORQUE_NM[THUMB_SERVO if is_thumb else FINGER_SERVO]
  stall *= SERVO_TORQUE_FRACTION
  pulses_per_rad = 4096 / (2 * math.pi)
  motor_kp = SERVO_POSITION_P_GAIN / 128 * pulses_per_rad / 885 * stall
  ratio = arm / PULLEY_RADIUS_M
  kp = motor_kp * ratio**2
  return kp, KV_OVER_KP * kp, stall * ratio


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
  if not xml_path.is_file():
    raise FileNotFoundError(f"RUKA hand MJCF not found: {xml_path}")
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _get_assets(xml_path, spec)
  return _apply_task_names(spec)


def get_my_hand_ruka_cfg() -> EntityCfg:
  """Build the 15-joint, 11-actuator RUKA hand entity for the reorient task."""
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.5),
      rot=(0.70710678, 0.0, -0.70710678, 0.0),
      # Coupled fingers: joint3 (tip) must start equal to joint2.
      joint_pos={
        ".*_finger1_joint1": 0.40,
        ".*_finger1_joint2": 0.30,
        ".*_finger1_joint3": 0.50,
        ".*_finger[2-5]_joint1": 0.60,
        ".*_finger[2-5]_joint[2-3]": 0.45,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=partial(_get_spec, _XML_PATH),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(target_names_expr=(".*_finger.*_joint.*",)),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
