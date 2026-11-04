"""Train recurrent PPO to rotate the cube without dropping it."""

from pathlib import Path

import numpy as np

from Environment.Environment import Environment
from PPO.ppo import PPO
from PPO.rollout import RolloutBuffer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "xml" / "hand_box_scene.xml"
SAVE_PATH = ROOT / "models" / "ppo_cube_rotation_lstm.pth"
POLICY_PATH = SAVE_PATH
LOG_PATH = ROOT / "training_log.txt"
TOTAL_EPISODES = 2000
# Training berjalan headless agar viewer MuJoCo tidak menghentikan loop PPO.
RENDER = True  # Aktifkan hanya saat ingin melihat viewer MuJoCo.


def log(message: str, log_file) -> None:
    """Print a message and save it immediately to the training log."""
    print(message)
    print(message, file=log_file, flush=True)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"File XML tidak ditemukan: {MODEL_PATH}")
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_file = LOG_PATH.open("w", encoding="utf-8")
    env = Environment(str(MODEL_PATH))
    try:
        initial_observation = env.reset()
        if RENDER:
            env.render()
        print(f"Total episode training : {TOTAL_EPISODES}")
        print(f"Maksimum langkah/episode: {env.MAX_STEPS}")
        ppo = PPO(
            observation_dim=initial_observation.size,
            action_dim=len(env.actuator_ids),
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            ppo_epochs=10,
        )
        if POLICY_PATH.is_file():
            ppo.load(str(POLICY_PATH))
            print(f"Policy rotasi dimuat: {POLICY_PATH}")
        reward_history = []
        survival_history = []
        best_reward = -np.inf
        best_model_score = (-1, -np.inf)

        for episode in range(1, TOTAL_EPISODES + 1):
            observation = env.reset()
            ppo.reset_hidden()
            if RENDER:
                env.render()
            buffer = RolloutBuffer()
            episode_reward = 0.0
            survived = False
            done = False

            while not done:
                action, log_prob, value = ppo.get_action(observation)
                next_observation, reward, done = env.step(action)
                survived = env.last_done_reason == "timeout"
                terminal = env.last_done_reason == "fallen"
                buffer.add(observation, action, reward, terminal, log_prob, value)
                observation = next_observation
                episode_reward += reward
                if RENDER and env.current_step % 5 == 0:
                    env.render()

            # Timeout is bootstrapped; successful termination is terminal.
            next_value = 0.0 if env.last_done_reason != "timeout" else ppo.get_value(observation)
            rollout = buffer.finish(next_value, ppo.gamma, ppo.gae_lambda)
            update_info = ppo.update(**rollout)

            reward_history.append(episode_reward)
            episode_survived = env.last_done_reason == "timeout"
            survival_history.append(int(episode_survived))
            model_score = (int(episode_survived), episode_reward)
            if model_score > best_model_score:
                best_reward = episode_reward
                best_model_score = model_score
                ppo.save(str(SAVE_PATH))

            if episode == 1 or episode % 10 == 0:
                log(
                    f"Episode {episode:4d} | Reward {episode_reward:8.2f} | "
                    f"Avg(10) {np.mean(reward_history[-10:]):8.2f} | "
                    f"Survival(10) {np.mean(survival_history[-10:]) * 100:5.1f}% | "
                    f"Episode steps {env.current_step:3d}/{env.MAX_STEPS} | "
                    f"Cube z {env.get_cube_position()[2]:7.3f} | "
                    f"Rotation {env.get_cube_rotation_speed():7.3f} | "
                    f"Done {env.last_done_reason:7s} | "
                    f"Actor {update_info['actor_loss']:8.4f} | "
                    f"Critic {update_info['critic_loss']:8.4f}",
                    log_file,
                )

        print(f"Training selesai pada episode {TOTAL_EPISODES}.")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Model: {SAVE_PATH}")
    finally:
        env.close()
        log_file.close()


if __name__ == "__main__":
    main()
