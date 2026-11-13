"""Plan a cube reorientation as a sequence of single +-90 deg tag-frame turns.

A hand without finger abduction may only master turns about some tag axes
(e.g. yaw and pitch). Any of the 24 face-aligned cube orientations is still
reachable from those turns alone, e.g. a roll quarter-turn is

    roll(+-90) = yaw(-90) -> pitch(+-90) -> yaw(+90)

(R_x(t) = R_y(90) R_z(t) R_y(-90), turns applied left to right in tag frame).
Only one direction per axis is needed as well: pitch(+90) = 3 x pitch(-90).

This module finds the shortest such sequence with a BFS over the 24 cube
rotations. Each step's quaternion is a goal in tag frame, exactly as the
training command samples them in ``cube_90`` mode (goal_new = R_turn *
goal_old), so the sub-goals can be fed to the policy one after another.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_mul

from wuji_mjlab.utils.math import cube_rotation_quats, nearest_cube_rotation_index

AXIS_NAMES = ("roll", "yaw", "pitch")
"""Tag axes 0/1/2 of the custom hand (x along fingers, y palm normal, z lateral)."""


@dataclass(frozen=True)
class CubeTurn:
  axis: int
  """Tag-frame axis 0/1/2."""
  sign: int
  """+1 for +90 deg, -1 for -90 deg."""
  goal_quat: tuple[float, float, float, float]
  """Cube orientation in tag frame (w, x, y, z) after this turn."""

  def label(self, axis_names: tuple[str, str, str] = AXIS_NAMES) -> str:
    return f"{axis_names[self.axis]}{'+' if self.sign > 0 else '-'}90"


def _turn_quat(axis: int, sign: int) -> torch.Tensor:
  q = torch.zeros(4)
  q[0] = math.sqrt(0.5)
  q[1 + axis] = sign * math.sqrt(0.5)
  return q


def _transition_table(
  cube_quats: torch.Tensor, turns: tuple[tuple[int, int], ...]
) -> dict[int, list[tuple[int, int, int]]]:
  """state -> [(next_state, axis, sign)] for every allowed quarter-turn."""
  table: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(24)}
  for axis, sign in turns:
    if axis not in (0, 1, 2) or sign not in (1, -1):
      raise ValueError(f"Invalid turn (axis={axis}, sign={sign}).")
    rotated = quat_mul(_turn_quat(axis, sign).expand(24, -1), cube_quats)
    dst = nearest_cube_rotation_index(rotated, cube_quats)
    for src in range(24):
      table[src].append((int(dst[src]), axis, sign))
  return table


def plan_cube_turns(
  start_quat: torch.Tensor | tuple[float, ...],
  goal_quat: torch.Tensor | tuple[float, ...],
  axes: tuple[int, ...] = (1, 2),
  turns: tuple[tuple[int, int], ...] | None = None,
) -> list[CubeTurn]:
  """Shortest list of +-90 deg turns taking start to goal.

  Allowed turns are both directions about each of ``axes``, or exactly the
  ``(axis, sign)`` pairs in ``turns`` when given, e.g. the directions the
  policy has mastered: ``turns=((1, 1), (1, -1), (2, -1))`` = yaw+-, pitch-.
  A missing direction is replaced by three turns the other way.

  Both orientations are cube-in-tag-frame quaternions (w, x, y, z) and are
  snapped to the nearest of the 24 face-aligned rotations. Returns [] when
  they snap to the same rotation. Raises ValueError when the allowed turns
  cannot reach the goal (only possible with turns about a single axis).
  """
  cube_quats = cube_rotation_quats("cpu")
  start = torch.as_tensor(start_quat, dtype=torch.float32).reshape(1, 4)
  goal = torch.as_tensor(goal_quat, dtype=torch.float32).reshape(1, 4)
  src = int(nearest_cube_rotation_index(start, cube_quats)[0])
  dst = int(nearest_cube_rotation_index(goal, cube_quats)[0])
  if src == dst:
    return []

  if turns is None:
    turns = tuple((axis, sign) for axis in axes for sign in (1, -1))
  table = _transition_table(cube_quats, tuple(turns))
  parent: dict[int, tuple[int, int, int]] = {}
  queue = deque([src])
  seen = {src}
  while queue:
    node = queue.popleft()
    for nxt, axis, sign in table[node]:
      if nxt in seen:
        continue
      seen.add(nxt)
      parent[nxt] = (node, axis, sign)
      if nxt == dst:
        queue.clear()
        break
      queue.append(nxt)
  if dst not in parent:
    raise ValueError(f"Goal is not reachable with turns {tuple(turns)}.")

  turns: list[CubeTurn] = []
  node = dst
  while node != src:
    prev, axis, sign = parent[node]
    q = cube_quats[node]
    turns.append(CubeTurn(axis, sign, tuple(float(v) for v in q)))
    node = prev
  return turns[::-1]


def single_turn_goal(
  start_quat: torch.Tensor | tuple[float, ...], axis: int, sign: int
) -> torch.Tensor:
  """Face-aligned goal one +-90 deg turn about ``axis`` away from start."""
  cube_quats = cube_rotation_quats("cpu")
  start = torch.as_tensor(start_quat, dtype=torch.float32).reshape(1, 4)
  src = cube_quats[nearest_cube_rotation_index(start, cube_quats)]
  return quat_mul(_turn_quat(axis, sign).reshape(1, 4), src)[0]
