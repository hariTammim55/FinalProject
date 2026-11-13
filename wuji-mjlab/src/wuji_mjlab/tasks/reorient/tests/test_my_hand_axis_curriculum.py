"""Axis curriculum of the reorient command and the quarter-turn planner."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from wuji_mjlab.tasks.reorient.cube_turn_planner import (
  plan_cube_turns,
  single_turn_goal,
)
from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)
from wuji_mjlab.utils.math import cube_rotation_quats

ROLL, YAW, PITCH = 0, 1, 2


class _FakeEntity:
  def __init__(self, num_envs, body_names, site_names=()):
    self._body_names = list(body_names)
    self._site_names = list(site_names)
    pose = torch.zeros(num_envs, max(len(site_names), 1), 7)
    pose[..., 3] = 1.0
    quat = torch.zeros(num_envs, 4)
    quat[:, 0] = 1.0
    qpos = torch.zeros(num_envs, 7)
    qpos[:, 3] = 1.0
    self.data = SimpleNamespace(
      body_link_pose_w=pose.clone(),
      site_pose_w=pose.clone(),
      root_link_quat_w=quat,
      root_link_pos_w=torch.zeros(num_envs, 3),
      data=SimpleNamespace(qpos=qpos),
    )
    self.indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7))

  def _find(self, names, pattern):
    regex = re.compile(pattern)
    ids = [i for i, n in enumerate(names) if regex.fullmatch(n)]
    return ids, [names[i] for i in ids]

  def find_bodies(self, pattern, preserve_order=False):
    return self._find(self._body_names, pattern)

  def find_sites(self, pattern, preserve_order=False):
    return self._find(self._site_names, pattern)


def _make_command(num_envs=64, **overrides) -> InHandReorientCommand:
  robot = _FakeEntity(num_envs, ["right_palm_link"], ["right_wrist_tag"])
  obj = _FakeEntity(num_envs, ["cube"])
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={"robot": robot, "object": obj},
    step_dt=0.05,
  )
  params = dict(
    resampling_time_range=(1.0e6, 1.0e6),
    entity_name="object",
    goal_mode="cube_90",
    goal_axes=(ROLL, YAW, PITCH),
    goal_axis_names=("roll", "yaw", "pitch"),
    success_threshold=0.4,
    success_hold_steps=2,
    goal_switch_delay=1000,
    axis_curriculum=True,
    axis_unlock_order=(YAW, PITCH, ROLL),
    axis_unlock_min_attempts=5.0,
    axis_rate_decay=1.0,
    axis_update_interval=1,
  )
  params.update(overrides)
  return InHandReorientCommand(InHandReorientCommandCfg(**params), env)


def _sampled_axes(cmd: InHandReorientCommand, rounds: int = 20) -> set[int]:
  ids = torch.arange(cmd.num_envs)
  axes: set[int] = set()
  for _ in range(rounds):
    cmd._sample_goal_in_world(ids)
    axes.update(cmd.goal_axis.tolist())
  return axes


class TestAxisCurriculum:
  def test_starts_with_first_axis_only(self):
    cmd = _make_command()
    assert cmd.active_axes == (YAW,)
    assert _sampled_axes(cmd) == {YAW}

  def test_both_signs_of_active_axis_are_sampled(self):
    cmd = _make_command()
    _sampled_axes(cmd)
    turns = set(cmd.goal_turn.tolist())
    for _ in range(10):
      cmd._sample_goal_in_world(torch.arange(cmd.num_envs))
      turns.update(cmd.goal_turn.tolist())
    assert turns == {2 * YAW, 2 * YAW + 1}

  def test_success_unlocks_next_axis_in_order(self):
    cmd = _make_command()
    _sampled_axes(cmd, rounds=1)
    # Hold the cube at every env's goal: each yaw turn succeeds.
    cmd.object.data.root_link_quat_w = cmd.goal_quat_w.clone()
    for _ in range(cmd.cfg.success_hold_steps):
      cmd._update_command()
    assert cmd.turn_reach_count[:, 2 * YAW : 2 * YAW + 2].sum() == cmd.num_envs
    assert cmd.active_axes == (YAW, PITCH)
    assert _sampled_axes(cmd) == {YAW, PITCH}

  def test_failing_turns_do_not_unlock(self):
    cmd = _make_command()
    _sampled_axes(cmd)  # attempts only, no successes
    for _ in range(5):
      cmd._update_command()
    assert cmd.active_axes == (YAW,)

  def test_failing_turn_is_sampled_more_often(self):
    cmd = _make_command(axis_min_weight=0.1)
    yaw_pos, yaw_neg = 2 * YAW, 2 * YAW + 1
    cmd._turn_attempts[yaw_pos] = cmd._turn_attempts[yaw_neg] = 100.0
    cmd._turn_successes[yaw_pos] = 100.0  # +90 mastered, -90 always fails
    cmd._refresh_turn_weights()
    assert cmd._turn_weight[yaw_neg] > 5 * cmd._turn_weight[yaw_pos]

  def test_per_sign_metrics_exist(self):
    cmd = _make_command()
    for name in ("roll", "yaw", "pitch"):
      for sign in ("pos", "neg"):
        assert f"reach_{name}_{sign}" in cmd.metrics
        assert f"curr_rate_{name}_{sign}" in cmd.metrics
    assert "curr_active_axes" in cmd.metrics

  def test_disabled_curriculum_samples_all_axes(self):
    cmd = _make_command(axis_curriculum=False)
    assert _sampled_axes(cmd) == {ROLL, YAW, PITCH}

  def test_unlock_order_must_match_goal_axes(self):
    with pytest.raises(ValueError, match="permutation"):
      _make_command(axis_unlock_order=(YAW, PITCH))

  def test_one_mastered_direction_unlocks_next_axis(self):
    cmd = _make_command(axis_initial_unlocked=2)
    assert cmd.active_axes == (YAW, PITCH)
    cmd._turn_attempts[:] = 100.0
    cmd._turn_successes[2 * YAW] = 90.0  # yaw+ mastered, yaw- not
    cmd._turn_successes[2 * PITCH + 1] = 80.0  # pitch- mastered, pitch+ fails
    cmd._update_command()
    assert cmd.active_axes == (YAW, PITCH, ROLL)

  def test_axis_without_any_mastered_direction_blocks_unlock(self):
    cmd = _make_command(axis_initial_unlocked=2)
    cmd._turn_attempts[:] = 100.0
    cmd._turn_successes[2 * YAW] = 90.0
    cmd._turn_successes[2 * PITCH : 2 * PITCH + 2] = 1.0
    cmd._update_command()
    assert cmd.active_axes == (YAW, PITCH)

  def test_failing_turn_is_given_up_only_after_grace(self):
    cmd = _make_command(axis_give_up_steps=10, axis_min_weight=0.2)
    yaw_pos, yaw_neg = 2 * YAW, 2 * YAW + 1
    cmd._turn_attempts[yaw_pos] = cmd._turn_attempts[yaw_neg] = 100.0
    cmd._turn_successes[yaw_pos] = 40.0  # below unlock threshold
    cmd._update_command()
    assert not cmd.given_up_turns.any()
    assert cmd._turn_weight[yaw_neg] > cmd._turn_weight[yaw_pos]
    for _ in range(10):
      cmd._update_command()
    assert cmd.given_up_turns.tolist() == [i == yaw_neg for i in range(6)]
    assert cmd._turn_weight[yaw_neg].item() == pytest.approx(0.2)
    assert "curr_given_up_yaw_neg" in cmd.metrics

  def test_given_up_turn_recovers_when_it_starts_succeeding(self):
    cmd = _make_command(axis_give_up_steps=0)
    yaw_neg = 2 * YAW + 1
    cmd._turn_attempts[yaw_neg] = 100.0
    cmd._update_command()
    assert cmd.given_up_turns[yaw_neg]
    cmd._turn_successes[yaw_neg] = 30.0
    cmd._update_command()
    assert not cmd.given_up_turns[yaw_neg]


class TestGoalTimeout:
  def test_unreached_goal_is_replaced_after_timeout(self):
    cmd = _make_command(axis_curriculum=False, goal_timeout_s=0.5)  # 10 steps
    ids = torch.arange(cmd.num_envs)
    cmd._sample_goal_in_world(ids)
    first_goal = cmd.goal_quat_w.clone()
    for _ in range(9):
      cmd._update_command()
    assert not cmd.goal_timed_out.any()
    assert torch.equal(cmd.goal_quat_w, first_goal)
    cmd._update_command()
    assert cmd.goal_timed_out.all()
    assert not cmd.goal_switched.any(), "a timeout is not a success switch"
    assert (cmd.goal_timer == 0).all()
    assert (cmd.goal_timeout_count == 1).all()
    assert cmd.goal_reach_count.sum() == 0

  def test_new_goal_is_one_turn_from_the_cube_not_the_failed_goal(self):
    cmd = _make_command(axis_curriculum=False, goal_timeout_s=0.5)
    cmd._sample_goal_in_world(torch.arange(cmd.num_envs))
    for _ in range(10):
      cmd._update_command()
    # Cube stayed at identity, so each new goal is a quarter-turn from it.
    err = quat_error_magnitude(cmd.goal_quat_w, cmd.object.data.root_link_quat_w)
    assert torch.allclose(err, torch.full_like(err, math.pi / 2), atol=1e-4)

  def test_timeout_disabled_by_default(self):
    cmd = _make_command(axis_curriculum=False)
    goal = cmd.goal_quat_w.clone()
    for _ in range(50):
      cmd._update_command()
    assert torch.equal(cmd.goal_quat_w, goal)
    assert "goal_timeouts" not in cmd.metrics


class TestCubeTurnPlanner:
  def test_roll_is_three_turns_with_yaw_and_pitch(self):
    identity = (1.0, 0.0, 0.0, 0.0)
    for sign in (1, -1):
      goal = single_turn_goal(identity, ROLL, sign)
      turns = plan_cube_turns(identity, goal, axes=(YAW, PITCH))
      assert len(turns) == 3
      assert {t.axis for t in turns} <= {YAW, PITCH}
      final = torch.tensor(turns[-1].goal_quat).reshape(1, 4)
      assert quat_error_magnitude(final, goal.reshape(1, 4)).item() < 1e-4

  def test_every_orientation_reachable_with_two_axes(self):
    cubes = cube_rotation_quats("cpu")
    for start in cubes:
      for goal in cubes:
        turns = plan_cube_turns(start, goal, axes=(YAW, PITCH))
        if turns:
          final = torch.tensor(turns[-1].goal_quat).reshape(1, 4)
          err = quat_error_magnitude(final, goal.reshape(1, 4)).item()
          assert err < 1e-4
        assert len(turns) <= 4

  def test_each_step_is_one_quarter_turn(self):
    start = torch.tensor([1.0, 0.0, 0.0, 0.0])
    goal = single_turn_goal(single_turn_goal(start, ROLL, 1), PITCH, -1)
    prev = start.reshape(1, 4)
    for turn in plan_cube_turns(start, goal, axes=(YAW, PITCH)):
      cur = torch.tensor(turn.goal_quat).reshape(1, 4)
      assert quat_error_magnitude(prev, cur).item() == pytest.approx(
        math.pi / 2, abs=1e-4
      )
      prev = cur

  def test_one_direction_per_axis_reaches_everything(self):
    # yaw+-, pitch- only (pitch+ never learned in the first run).
    turns = ((YAW, 1), (YAW, -1), (PITCH, -1))
    cubes = cube_rotation_quats("cpu")
    for start in cubes:
      for goal in cubes:
        plan = plan_cube_turns(start, goal, turns=turns)
        assert all((t.axis, t.sign) in turns for t in plan)
        if plan:
          final = torch.tensor(plan[-1].goal_quat).reshape(1, 4)
          assert quat_error_magnitude(final, goal.reshape(1, 4)).item() < 1e-4

  def test_missing_direction_becomes_three_turns_the_other_way(self):
    identity = (1.0, 0.0, 0.0, 0.0)
    goal = single_turn_goal(identity, PITCH, 1)
    plan = plan_cube_turns(identity, goal, turns=((YAW, 1), (PITCH, -1)))
    assert [t.label() for t in plan] == ["pitch-90"] * 3

  def test_single_axis_cannot_reach_other_axes(self):
    identity = (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="not reachable"):
      plan_cube_turns(identity, single_turn_goal(identity, ROLL, 1), axes=(YAW,))
