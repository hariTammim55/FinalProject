import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class Environment(gym.Env):
    """
    Custom Gymnasium environment untuk robot hand Wuji Lab.

    Task:
        Mengontrol posisi joint robot hand menuju target sudut.

    Action:
        Target posisi actuator yang dinormalisasi ke [-1, 1].

    Observation:
        qpos   : posisi seluruh generalized coordinates.
        qvel   : kecepatan seluruh generalized coordinates.
        target : target posisi joint yang dikontrol.

    Catatan:
        Implementasi ini mengasumsikan actuator yang digunakan
        adalah actuator joint dengan kontrol posisi.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path,
        frame_skip=5,
        max_episode_steps=500,
        success_threshold=0.05,
        render_mode=None,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. LOAD MUJOCO MODEL
        # --------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.render_mode = render_mode

        self.viewer = None
        self.steps = 0

        # --------------------------------------------------
        # 2. DETEKSI ACTUATOR YANG MENGONTROL JOINT
        # --------------------------------------------------
        self.actuator_ids = np.arange(
            self.model.nu, dtype=np.int32
        )

        if self.model.nu == 0:
            raise ValueError(
                "Model MuJoCo tidak memiliki actuator."
            )

        self.joint_ids = []

        for actuator_id in self.actuator_ids:
            # Joint yang dikontrol actuator.
            joint_id = int(
                self.model.actuator_trnid[actuator_id, 0]
            )

            if joint_id < 0:
                raise ValueError(
                    f"Actuator {actuator_id} bukan actuator "
                    "dengan transmisi joint langsung."
                )

            joint_type = int(self.model.jnt_type[joint_id])

            supported_types = (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            )

            if joint_type not in supported_types:
                raise ValueError(
                    f"Joint pada actuator {actuator_id} "
                    "bukan joint hinge atau slide."
                )

            # Setiap actuator harus mengontrol joint berbeda.
            if joint_id in self.joint_ids:
                raise ValueError(
                    "Beberapa actuator mengontrol joint yang sama. "
                    "Perlu pemetaan kontrol khusus."
                )

            self.joint_ids.append(joint_id)

        self.joint_ids = np.asarray(
            self.joint_ids, dtype=np.int32
        )

        # Alamat posisi dan kecepatan setiap joint.
        self.qpos_ids = self.model.jnt_qposadr[
            self.joint_ids
        ].astype(np.int32)

        self.dof_ids = self.model.jnt_dofadr[
            self.joint_ids
        ].astype(np.int32)

        # --------------------------------------------------
        # 3. RENTANG ACTION DAN TARGET JOINT
        # --------------------------------------------------
        # Actuator harus memiliki ctrlrange yang valid.
        if not np.all(self.model.actuator_ctrllimited):
            raise ValueError(
                "Semua actuator harus memiliki ctrlrange "
                "yang terbatas untuk environment ini."
            )

        self.ctrl_low = self.model.actuator_ctrlrange[
            :, 0
        ].astype(np.float64)

        self.ctrl_high = self.model.actuator_ctrlrange[
            :, 1
        ].astype(np.float64)

        if np.any(self.ctrl_low >= self.ctrl_high):
            raise ValueError(
                "Rentang actuator ctrlrange tidak valid."
            )

        # Action dinormalisasi agar mudah digunakan algoritma RL.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Target joint diambil dari irisan rentang joint
        # dan rentang kontrol actuator.
        goal_low = self.ctrl_low.copy()
        goal_high = self.ctrl_high.copy()

        for i, joint_id in enumerate(self.joint_ids):
            if self.model.jnt_limited[joint_id]:
                joint_low, joint_high = (
                    self.model.jnt_range[joint_id]
                )

                goal_low[i] = max(
                    goal_low[i], joint_low
                )
                goal_high[i] = min(
                    goal_high[i], joint_high
                )

        if np.any(goal_low >= goal_high):
            raise ValueError(
                "Rentang target joint dan actuator "
                "tidak memiliki irisan yang valid."
            )

        self.goal_low = goal_low
        self.goal_high = goal_high

        # --------------------------------------------------
        # 4. OBSERVATION SPACE
        # --------------------------------------------------
        # Observation = qpos + qvel + target joint.
        obs_size = (
            self.model.nq
            + self.model.nv
            + len(self.joint_ids)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.goal = np.zeros(
            len(self.joint_ids), dtype=np.float64
        )

        print("WujiHandEnv berhasil dibuat.")
        print(f"Jumlah actuator : {self.model.nu}")
        print(f"Jumlah joint    : {len(self.joint_ids)}")
        print(f"Observation dim : {obs_size}")

    # ------------------------------------------------------
    # 5. OBSERVATION
    # ------------------------------------------------------
    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.goal.copy(),
            ]
        )

        return obs.astype(np.float32)

    # ------------------------------------------------------
    # 6. RESET ENVIRONMENT
    # ------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.steps = 0

        # Sampel target posisi joint baru.
        self.goal = self.np_random.uniform(
            low=self.goal_low,
            high=self.goal_high,
        )

        # Hitung kondisi awal.
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()

        info = {
            "goal": self.goal.copy(),
            "joint_position": self.data.qpos[
                self.qpos_ids
            ].copy(),
        }

        return observation, info

    # ------------------------------------------------------
    # 7. STEP ENVIRONMENT
    # ------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Shape action harus "
                f"{self.action_space.shape}, "
                f"tetapi menerima {action.shape}."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                "Action mengandung NaN atau infinity."
            )

        # Pastikan action berada dalam rentang yang valid.
        action = np.clip(action, -1.0, 1.0)

        # --------------------------------------------------
        # 7A. DENORMALISASI ACTION
        # --------------------------------------------------
        # Action [-1, 1] menjadi target kontrol actuator.
        ctrl = self.ctrl_low + (
            (action + 1.0) / 2.0
        ) * (self.ctrl_high - self.ctrl_low)

        self.data.ctrl[:] = ctrl

        # --------------------------------------------------
        # 7B. JALANKAN SIMULASI MUJOCO
        # --------------------------------------------------
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        # --------------------------------------------------
        # 7C. HITUNG ERROR TARGET
        # --------------------------------------------------
        joint_position = self.data.qpos[
            self.qpos_ids
        ].copy()

        error = joint_position - self.goal

        # Error posisi joint yang dinormalisasi.
        joint_range = self.goal_high - self.goal_low

        normalized_error = error / joint_range

        # --------------------------------------------------
        # 7D. HITUNG REWARD
        # --------------------------------------------------
        tracking_error = np.mean(
            np.square(normalized_error)
        )

        # Reward tinggi jika joint mendekati target.
        reward = -float(tracking_error)

        # Bonus ketika seluruh target tercapai.
        success = bool(
            np.max(np.abs(error)) < self.success_threshold
        )

        if success:
            reward += 1.0

        # --------------------------------------------------
        # 7E. TERMINATION DAN TRUNCATION
        # --------------------------------------------------
        terminated = success

        truncated = (
            self.steps >= self.max_episode_steps
        )

        # Pemeriksaan kondisi numerik simulasi.
        if not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
        ):
            raise RuntimeError(
                "Simulasi menghasilkan posisi atau "
                "kecepatan non-finite."
            )

        # --------------------------------------------------
        # 7F. RETURN GYMNASIUM
        # --------------------------------------------------
        observation = self._get_obs()

        info = {
            "goal": self.goal.copy(),
            "joint_position": joint_position,
            "joint_error": error.copy(),
            "is_success": success,
        }

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------
    # 8. RENDER SIMULASI
    # ------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(
                self.model,
                self.data,
            )

        self.viewer.sync()

    # ------------------------------------------------------
    # 9. CLOSE ENVIRONMENT
    # ------------------------------------------------------
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None