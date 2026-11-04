"""MJLab adapter for the custom hand MJCF."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


_XML_PATH = Path(__file__).resolve().parent / "mjcf" / "hand_assembly.xml"
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
  for exclude in spec.excludes:
    exclude.bodyname1 = body_names[exclude.bodyname1]
    exclude.bodyname2 = body_names[exclude.bodyname2]
  for joint in spec.joints:
    joint.name = joint_names[joint.name]
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


def _get_spec(xml_path: Path) -> mujoco.MjSpec:
  if not xml_path.is_file():
    raise FileNotFoundError(f"Custom hand MJCF not found: {xml_path}")
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _get_assets(xml_path, spec)
  return _apply_task_names(spec)


def get_my_hand_cfg() -> EntityCfg:
  """Build the 15-DOF custom hand entity for the reorient task."""
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.5),
      rot=(0.70710678, 0.0, -0.70710678, 0.0),
      joint_pos={
        ".*_finger1_joint1": 0.40,
        ".*_finger1_joint2": 0.30,
        ".*_finger1_joint3": 0.50,
        ".*_finger[2-5]_joint1": 0.60,
        ".*_finger[2-5]_joint2": 0.40,
        ".*_finger[2-5]_joint3": 0.50,
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