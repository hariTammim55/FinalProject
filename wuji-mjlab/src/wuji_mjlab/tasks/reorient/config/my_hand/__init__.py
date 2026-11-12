"""Custom hand reorient task registration."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

from .env_cfgs import my_hand_reorient_env_cfg


def my_hand_reorient_ppo_runner_cfg(
  run_name: str = "MyHand_Reorient", max_iterations: int = 5000
) -> RslRlOnPolicyRunnerCfg:
  """Wuji's PPO cfg with more sustained exploration for the custom hand.

  With Wuji's settings the action std collapsed to its 0.2 floor within ~150
  iterations, before the policy ever found how to rotate the cube.
  """
  cfg = wuji_hand_reorient_ppo_runner_cfg(
    run_name=run_name, max_iterations=max_iterations
  )
  cfg.actor.distribution_cfg = {
    **cfg.actor.distribution_cfg,
    "std_range": (0.35, 50.0),
  }
  cfg.algorithm.entropy_coef = 0.005
  return cfg


register_mjlab_task(
  task_id="WujiMyHand_Reorient",
  env_cfg=my_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=my_hand_reorient_env_cfg(play=True),
  rl_cfg=my_hand_reorient_ppo_runner_cfg(),
  runner_cls=WujiOnPolicyRunner,
)
