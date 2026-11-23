"""Play a trained MyHand_Reorient policy for ONE chosen cube turn.

The normal play mode gives the policy a random +-90 deg goal (any axis) every
time it reaches one. Here every goal is the turn you pick: after each success
the next goal is the same turn again, so the cube keeps rotating that way
(e.g. roll +90, +90, +90, ...). With --once the turn is done once per episode
and the hand then holds the cube. A new episode starts when the cube drops or
after the 60 s time limit.

Usage (from the repo root):
  pixi run python scripts/play_single_turn.py --turn yaw+90
  pixi run python scripts/play_single_turn.py --turn roll-90 --viewer viser
  pixi run python scripts/play_single_turn.py --turn pitch+90 \
      --checkpoint logs/rsl_rl/wuji_reorient/<run>/model_XXXX.pt

Turns are about the cube-tag axes used in training: roll, yaw, pitch.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

import wuji_mjlab.tasks  # noqa: F401  (registers MyHand_Reorient)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_mul
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.my_hand.env_cfgs import (
  MY_HAND_TAG_PITCH_AXIS,
  MY_HAND_TAG_ROLL_AXIS,
  MY_HAND_TAG_YAW_AXIS,
)
from wuji_mjlab.tasks.reorient.mdp.commands import InHandReorientCommand
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK_ID = "MyHand_Reorient"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
  REPO_ROOT
  / "logs/rsl_rl/wuji_reorient/2026-11-16_08-10-15_MyHand_Reorient/model_9996.pt"
)
AXES = {"roll": MY_HAND_TAG_ROLL_AXIS, "yaw": MY_HAND_TAG_YAW_AXIS, "pitch": MY_HAND_TAG_PITCH_AXIS}
# Command turn index = 2 * axis + (0 for +90 deg, 1 for -90 deg).
TURNS = {
  f"{name}{sign}90": 2 * axis + (0 if sign == "+" else 1)
  for name, axis in AXES.items()
  for sign in ("+", "-")
}


def install_single_turn(turn: int, once: bool = False) -> None:
  """Patch the goal command so every goal is the chosen turn.

  once=False: after each success the next goal is the same turn again (the
  cube keeps rotating the same way). once=True: one goal per episode, then hold.
  """
  orig_resample = InHandReorientCommand._resample_command

  def _resample_command(self, env_ids):
    if not hasattr(self, "_single_turn_pending"):
      self._single_turn_pending = torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device
      )
    self._single_turn_pending[env_ids] = True  # new episode -> new goal
    orig_resample(self, env_ids)

  def _sample_goal_in_world(self, env_ids):
    # Called on reset, after a success and after a goal timeout (the reference
    # goal_cube_idx is then the reached goal / the cube's snapped pose).
    # once=True: only the reset call sets a goal; afterwards the goal is kept.
    ids = env_ids[self._single_turn_pending[env_ids]] if once else env_ids
    if ids.numel() == 0:
      return
    self._single_turn_pending[ids] = False
    ref_idx = self.goal_cube_idx[ids]
    match = self._cube_neighbor_turn[ref_idx] == turn  # (n, 24), one True per row
    new_idx = match.float().argmax(dim=-1)
    self.goal_cube_idx[ids] = new_idx
    self.goal_axis[ids] = self._cube_neighbor_axis[ref_idx, new_idx]
    self.goal_turn[ids] = turn
    self.axis_goal_count[ids, turn // 2] += 1.0
    self.turn_goal_count[ids, turn] += 1.0
    tag_quat_w = self.robot.data.site_pose_w[ids, self.tag_site_id, 3:7]
    self.goal_quat_w[ids] = quat_mul(tag_quat_w, self._cube_quats[new_idx])

  InHandReorientCommand._resample_command = _resample_command
  InHandReorientCommand._sample_goal_in_world = _sample_goal_in_world


def install_wrong_way_replan(replan_deg: float, replan_s: float) -> None:
  """Re-plan a goal as soon as the cube was turned the wrong way.

  Sometimes (~0.3% of episodes, at the start) the policy turns the cube the
  opposite way, leaving it 180 deg from the goal. Training only ever showed
  90 deg goals, so it then stalls until the 15 s goal timeout. Here, once the
  error stays above replan_deg for replan_s, the reference snaps to the cube's
  current face and the same turn is asked again from there.
  """
  if replan_deg <= 0:
    return
  orig_update = InHandReorientCommand._update_command
  thr = math.radians(replan_deg)

  def _update_command(self, env_ids=None):
    orig_update(self, env_ids)
    if not hasattr(self, "_wrong_way_steps"):
      self._wrong_way_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self.replan_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    need = max(int(round(replan_s / self._env.step_dt)), 1)
    err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    self._wrong_way_steps = torch.where(err > thr, self._wrong_way_steps + 1, 0)
    ids = (self._wrong_way_steps >= need).nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    self._single_turn_pending[ids] = True  # also lets --once set a new goal
    self._snap_reference_to_cube(ids)
    self._sample_goal_in_world(ids)
    self.goal_timer[ids] = 0
    self.hold_counter[ids] = 0
    self._wrong_way_steps[ids] = 0
    self.replan_count[ids] += 1

  InHandReorientCommand._update_command = _update_command


class StatusPolicy:
  """Wraps the policy and prints when env 0 starts / finishes its turn."""

  def __init__(self, policy, env: ManagerBasedRlEnv, turn_name: str, once: bool):
    self.policy, self.env, self.turn_name, self.once = policy, env, turn_name, once
    self.cmd = env.command_manager.get_term("reorient_command")
    self.episode = 0
    self.reached = 0
    self.last_step = -1
    self.last_success_t = 0.0
    self.replans = 0

  def __call__(self, obs):
    step = int(self.env.episode_length_buf[0])
    if step < self.last_step or self.last_step < 0:
      self.episode += 1
      self.reached = 0
      self.last_success_t = 0.0
      print(f"[episode {self.episode}] goal: {self.turn_name}")
    self.last_step = step
    count = int(self.cmd.goal_reach_count[0])
    if count > self.reached and not (self.once and self.reached > 0):
      t = step * self.env.step_dt
      print(
        f"[episode {self.episode}] {self.turn_name} #{count} reached at {t:.1f} s "
        f"(+{t - self.last_success_t:.1f} s)" + (", holding" if self.once else "")
      )
      self.last_success_t = t
    self.reached = count
    replans = int(getattr(self.cmd, "replan_count", torch.zeros(1))[0])
    if replans > self.replans:
      print(f"[episode {self.episode}] cube turned the wrong way, new {self.turn_name} goal from its current face")
    self.replans = replans
    return self.policy(obs)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--turn", required=True, choices=sorted(TURNS), help="cube turn to perform")
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument("--once", action="store_true", help="do the turn once per episode, then hold the cube")
  p.add_argument("--replan-deg", type=float, default=150.0,
                 help="re-plan when the cube error stays above this (wrong-way turn); 0 = off")
  p.add_argument("--replan-s", type=float, default=1.0, help="seconds above --replan-deg before re-planning")
  p.add_argument("--goal-timeout", type=float, default=5.0,
                 help="re-plan from the cube's nearest face if a turn is not done in this many s "
                 "(training uses 15; a normal turn takes ~2 s)")
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = p.parse_args()

  if not args.checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
  install_single_turn(TURNS[args.turn], once=args.once)
  install_wrong_way_replan(args.replan_deg, args.replan_s)

  env_cfg, agent_cfg = prepare_task_cfgs(TASK_ID, [], play=True)
  env_cfg.scene.num_envs = args.num_envs
  # A stalled turn (cube stuck just outside the success threshold, or halfway)
  # is re-planned from the cube's nearest face after this timeout.
  env_cfg.commands["reorient_command"].goal_timeout_s = args.goal_timeout
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = WujiOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
  print(f"[INFO] checkpoint: {args.checkpoint}")
  policy = StatusPolicy(runner.get_inference_policy(device=args.device), env, args.turn, args.once)

  viewer = args.viewer
  if viewer == "auto":
    viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  (NativeMujocoViewer if viewer == "native" else ViserPlayViewer)(wrapped, policy).run()
  wrapped.close()


if __name__ == "__main__":
  main()
