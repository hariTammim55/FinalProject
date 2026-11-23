"""RUKA-actuated custom hand: 11 servos, DIP coupled to PIP, servo gains."""

import mujoco
import numpy as np
import pytest

from wuji_mjlab.assets.robots.MyHand import my_hand_cfg as hc


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  m = hc._get_spec(hc._XML_PATH).compile()
  m.opt.timestep = 0.01  # task timestep
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return m


def _target(m: mujoco.MjModel, i: int) -> str:
  return m.joint(m.actuator_trnid[i, 0]).name


def test_eleven_actuators_no_tip_joint(model):
  targets = [_target(model, i) for i in range(model.nu)]
  assert model.nu == 11
  for f in range(2, 6):
    assert f"right_finger{f}_joint1" in targets
    assert f"right_finger{f}_joint2" in targets
    assert f"right_finger{f}_joint3" not in targets
  for j in (1, 2, 3):
    assert f"right_finger1_joint{j}" in targets


def test_task_joint_names_and_ranges(model):
  """15 task joints, fingers 2-5 first then the thumb, flexion-only ranges.

  Same joints and limits as the removed 15-actuator myHand model.
  """
  names = [model.joint(i).name for i in range(model.njnt)]
  expected = [
    f"right_finger{f}_joint{j}" for f in (2, 3, 4, 5, 1) for j in (1, 2, 3)
  ]
  assert names == expected
  upper_deg = {
    "right_finger1_joint1": 189.87821331,  # thumb CMC
    "right_finger1_joint2": 89.95437384,  # thumb MCP
    "right_finger1_joint3": 119.74817918,  # thumb IP
  }
  for f in range(2, 6):
    upper_deg[f"right_finger{f}_joint1"] = 139.92889864  # MCP
    upper_deg[f"right_finger{f}_joint2"] = 119.74817918
    upper_deg[f"right_finger{f}_joint3"] = 119.74817918
  for i, name in enumerate(names):
    np.testing.assert_allclose(
      model.jnt_range[i], (0.0, np.radians(upper_deg[name])), atol=1e-6
    )


def test_tip_joint_follows_middle_joint(model):
  assert model.neq == 4
  d = mujoco.MjData(model)
  d.ctrl[:] = 0.4 * model.actuator_ctrlrange[:, 1]
  for _ in range(100):
    mujoco.mj_step(model, d)
  assert np.isfinite(d.qpos).all()
  for f in range(2, 6):
    middle = d.joint(f"right_finger{f}_joint2").qpos[0]
    tip = d.joint(f"right_finger{f}_joint3").qpos[0]
    assert middle > 0.3
    assert abs(middle - tip) < 0.01


def test_xml_gains_match_servo_model(model):
  """kp/kv/forcerange written in the MJCF == servo_actuator_params."""
  for i in range(model.nu):
    kp, kv, limit = hc.servo_actuator_params(_target(model, i))
    assert model.actuator_gainprm[i, 0] == pytest.approx(kp, rel=1e-3)
    assert model.actuator_biasprm[i, 1] == pytest.approx(-kp, rel=1e-3)
    assert model.actuator_biasprm[i, 2] == pytest.approx(-kv, rel=1e-3)
    assert model.actuator_forcelimited[i]
    assert model.actuator_forcerange[i, 1] == pytest.approx(limit, rel=1e-3)
    assert model.actuator_forcerange[i, 0] == pytest.approx(-limit, rel=1e-3)


def test_tendon_torque_limits():
  _, _, finger_mcp = hc.servo_actuator_params("right_finger2_joint1")
  _, _, thumb_mcp = hc.servo_actuator_params("right_finger1_joint2")
  f = hc.SERVO_TORQUE_FRACTION
  assert finger_mcp == pytest.approx(
    3.0 * f / hc.PULLEY_RADIUS_M * hc.TENDON_MOMENT_ARM_M["finger_MCP"]
  )
  assert thumb_mcp == pytest.approx(
    4.1 * f / hc.PULLEY_RADIUS_M * hc.TENDON_MOMENT_ARM_M["thumb_MCP"]
  )
