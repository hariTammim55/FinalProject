"""MyHand_Reorient task registration (custom RUKA-design hand, 11 servos)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

from .env_cfgs import my_hand_reorient_env_cfg


def my_hand_reorient_ppo_runner_cfg(
  run_name: str = "MyHand_Reorient", max_iterations: int = 6000
) -> RslRlOnPolicyRunnerCfg:
  """Wuji's PPO cfg with more sustained exploration for the custom hand.

  With Wuji's settings the action std collapsed to its 0.2 floor within ~150
  iterations, before the policy ever found how to rotate the cube.
  """
  cfg = wuji_hand_reorient_ppo_runner_cfg(
    run_name=run_name, max_iterations=max_iterations
  )
  # Floor 0.35 keeps exploring; the 1.0 cap (was 50) and the lower entropy
  # bonus (was 0.005) stop the std drift seen in the first 5000-iter run: std
  # rose 0.36 -> 0.55 after iter ~3750, action_rate grew 2.5x and the mean
  # reward collapsed from ~+40 to -270 while reach counts stayed flat.
  cfg.actor.distribution_cfg = {
    **cfg.actor.distribution_cfg,
    "std_range": (0.35, 1.0),
  }
  cfg.algorithm.entropy_coef = 0.002
  # A quarter-turn takes several seconds of coordinated finger motion; a
  # longer horizon (~200 steps = 10 s at 20 Hz) credits the whole manoeuvre.
  cfg.algorithm.gamma = 0.995
  return cfg


register_mjlab_task(
  task_id="MyHand_Reorient",
  # 4096 envs fit a 16 GB GPU (8192 needs ~19.5 GB).
  env_cfg=my_hand_reorient_env_cfg(num_envs=4096),
  play_env_cfg=my_hand_reorient_env_cfg(play=True),
  # New log folders end in "_MyHand_Reorient"; runs before the rename end in
  # "_MyHandRuka_Reorient" (same robot and task, checkpoints load as-is).
  rl_cfg=my_hand_reorient_ppo_runner_cfg(),
  runner_cls=WujiOnPolicyRunner,
)
