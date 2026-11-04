"""Custom hand reorient task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

from .env_cfgs import my_hand_reorient_env_cfg

register_mjlab_task(
  task_id="WujiMyHand_Reorient",
  env_cfg=my_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=my_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="MyHand_Reorient", max_iterations=5000
  ),
  runner_cls=WujiOnPolicyRunner,
)