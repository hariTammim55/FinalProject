import os
import time

from Environment.Environment import Environment
from PPO.ppo import PPO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "assets/xml/hand_box_scene.xml"
PPO_MODEL_PATH = "models/ppo_cube_rotation_lstm.pth"

# Jumlah episode pengujian
TEST_EPISODES = 10

# Delay visualisasi
RENDER_DELAY = 1
RENDER = True


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check XML
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File XML tidak ditemukan:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check trained model
    # --------------------------------------------------------

    if not os.path.exists(PPO_MODEL_PATH):
        raise FileNotFoundError(
            f"Model PPO tidak ditemukan:\n{PPO_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = Environment(
        MODEL_PATH
    )
    observation = env.reset()

    # --------------------------------------------------------
    # Create PPO
    # --------------------------------------------------------

    ppo = PPO(
        observation_dim=observation.size,
        action_dim=len(env.actuator_ids),
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    ppo.load(
        PPO_MODEL_PATH
    )

    print("\n========================================")
    print("             PPO TESTING")
    print("========================================")
    print(f"Model : {PPO_MODEL_PATH}")
    print(f"Episodes : {TEST_EPISODES}")
    print("========================================\n")

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    for episode in range(
        1,
        TEST_EPISODES + 1
    ):

        observation = env.reset()
        ppo.reset_hidden()
        if RENDER:
            env.render()

        done = False
        total_reward = 0.0
        step = 0

        print(
            f"\nEpisode {episode}"
        )

        while not done:

            # ------------------------------------------------
            # Get action from trained PPO
            # ------------------------------------------------

            action, _, _ = (
                ppo.get_action(
                    observation
                )
            )

            # ------------------------------------------------
            # Step environment
            # ------------------------------------------------

            observation, reward, done = (
                env.step(action)
            )

            total_reward += reward
            step += 1

            # ------------------------------------------------
            # Render MuJoCo
            # ------------------------------------------------

            if RENDER:
                env.render()

            # ------------------------------------------------
            # Print cube pitch
            # ------------------------------------------------

            if step % 10 == 0:

                pitch = env.get_cube_pitch()

                error = abs(
                    env.TARGET_PITCH_DEG
                    - pitch
                )

                print(
                    f"\rStep: {step:3d} | "
                    f"Cube z: {env.get_cube_position()[2]:7.3f} | "
                    f"Rotation: {env.get_cube_rotation_speed():7.3f} | "
                    f"Reward: {total_reward:8.2f}",
                    end="",
                )

            if RENDER:
                time.sleep(RENDER_DELAY)

        # ----------------------------------------------------
        # Episode result
        # ----------------------------------------------------

        survived = env.last_done_reason == "timeout"

        print()

        print(
            f"Episode {episode} selesai:"
        )

        print(
            f"  Steps       : {step}"
        )

        print(
            f"  Total reward: {total_reward:.2f}"
        )

        print(
            f"  Cube z      : {env.get_cube_position()[2]:.3f}"
        )

        print(
            f"  Rotation    : {env.get_cube_rotation_speed():.3f}"
        )

        print(
            f"  Survived    : {'YES' if survived else 'NO'}"
        )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    env.close()

    print(
        "\nTesting selesai."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()