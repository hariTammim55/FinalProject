"""MJLab adapter for the custom hand with RUKA tendon actuation (11 servos).

Actuated like the real RUKA-design hand: each finger has one servo on MCP
and ONE tendon pulling PIP and DIP together (DIP follows PIP through a joint
equality), the thumb has one servo per joint. Gains and torque limits in
``mjcf/hand_assembly.xml`` come from the Dynamixel servos, see
``servo_actuator_params``.

Joint / body names are converted to the task convention by
``_apply_task_names``, so the task sees right_finger<f>_joint<1..3>; the
policy has 11 actions (no action for right_finger[2-5]_joint3).
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


_XML_PATH = Path(__file__).resolve().parent / "mjcf" / "hand_assembly.xml"

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


# XML name -> task name: finger number (1 = thumb) and link names per joint.
_FINGER_PREFIXES = {
  "Index": 2,
  "Middle": 3,
  "Ring": 4,
  "Pinky": 5,
  "Thumb": 1,
}
_FINGER_LINK_NAMES = {
  "Index": ("MCP", "DIP", "PIP"),
  "Middle": ("MCP", "DIP", "PIP"),
  "Ring": ("MCP", "DIP", "PIP"),
  "Pinky": ("MCP", "DIP", "PIP"),
  "Thumb": ("CMC", "MCP", "IP"),
}
# Keyed by joint-name fragment; the first fragment found in the joint name
# wins, so the thumb-specific "Thumb_MCP" must come before the generic "MCP"
# (thumb MCP is 90 deg, matching its 0..1.57 rad actuator ctrlrange).
_JOINT_LIMITS_DEG = {
  "Thumb_MCP": 89.95437384,
  "MCP": 139.92889864,
  "DIP": 119.74817918,
  "PIP": 119.74817918,
  "CMC": 189.87821331,
  "IP": 119.74817918,
}


def _get_assets(xml_path: Path, spec: mujoco.MjSpec) -> dict[str, bytes]:
  mesh_dir = (xml_path.parent / spec.meshdir).resolve()
  return {
    f"{spec.meshdir}/{mesh.file}" if spec.meshdir else mesh.file: (
      mesh_dir / mesh.file
    ).read_bytes()
    for mesh in spec.meshes
  }


def _apply_task_names(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  body_names: dict[str, str] = {"Palm_Link": "right_palm_link"}
  joint_names: dict[str, str] = {}
  for finger_name, finger_number in _FINGER_PREFIXES.items():
    for link_number, link_name in enumerate(
      _FINGER_LINK_NAMES[finger_name], start=2
    ):
      body_names[f"{finger_name}_{link_name}_Link"] = (
        f"right_finger{finger_number}_link{link_number}"
      )
      joint_names[f"{finger_name}_{link_name}_Joint"] = (
        f"right_finger{finger_number}_joint{link_number - 1}"
      )

  for actuator in spec.actuators:
    actuator.target = joint_names[actuator.target]
  for equality in spec.equalities:
    if equality.type == mujoco.mjtEq.mjEQ_JOINT:
      equality.name1 = joint_names[equality.name1]
      if equality.name2:
        equality.name2 = joint_names[equality.name2]
  for exclude in spec.excludes:
    exclude.bodyname1 = body_names[exclude.bodyname1]
    exclude.bodyname2 = body_names[exclude.bodyname2]
  for joint in spec.joints:
    joint_type = next(
      part for part in _JOINT_LIMITS_DEG if f"{part}_Joint" in joint.name
    )
    joint.range = (0.0, _JOINT_LIMITS_DEG[joint_type])
    joint.name = joint_names[joint.name]
    joint.limited = True
  for body in spec.bodies:
    if body.name in body_names:
      body.name = body_names[body.name]

  for geom in spec.geoms:
    if geom.parent.name == "right_palm_link" and geom.contype:
      geom.name = "right_palm_collision"
    elif geom.contype or geom.conaffinity:
      geom.name = f"{geom.parent.name}_col"
    if geom.contype or geom.conaffinity:
      geom.margin = 0.0

  for finger_number in range(1, 6):
    spec.body(f"right_finger{finger_number}_link4").add_site(
      name=f"right_finger{finger_number}_tip",
      pos=(-0.028, 0.0, 0.0),
      group=4,
    )
  return spec


def servo_actuator_params(joint_name: str) -> tuple[float, float, float]:
  """(kp [N·m/rad], kv [N·m·s/rad], torque limit [N·m]) for a driven joint.

  ``joint_name`` is a task joint name (right_finger<f>_joint<n>). The servo
  sees the joint through the tendon: joint torque = tendon force * arm,
  tendon force = motor torque / spool radius, and a joint motion dq turns the
  spool by dq * arm / radius, so the joint-side stiffness is the servo
  stiffness times (arm / radius)^2. The values written in the MJCF must
  match these (checked by tests/test_my_hand_actuation.py).
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


def get_my_hand_cfg() -> EntityCfg:
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
