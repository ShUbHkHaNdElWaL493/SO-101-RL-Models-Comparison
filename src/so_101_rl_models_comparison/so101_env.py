import gymnasium as gym
import mujoco
import numpy as np

class SO101Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, mjcf_file, mode='train', render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(mjcf_file)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # Action Space: 6 continuous values for 5 arm joints + 1 gripper
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space: 21 dimensions total
        # - Current State (13 dims): 6 joint pos, 3 EE pos, 4 EE quat
        # - Goal State (8 dims): 3 EE pos, 4 EE quat, 1 gripper state
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.frame_skip = 50
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.target_gripper = 0.0

    def _get_obs(self):
        arm_qpos = self.data.qpos[:6]
        ee_pos = self.data.body("gripper").xpos.copy()
        ee_quat = self.data.body("gripper").xquat.copy()
        current_state = np.concatenate([arm_qpos, ee_pos, ee_quat])

        target_state = np.concatenate([
            self.target_pos, 
            self.target_quat, 
            np.array([self.target_gripper], dtype=np.float32)
        ])

        return np.concatenate([current_state, target_state]).astype(np.float32)

    def _sample_target(self):
        self.target_pos = np.array([
            np.random.uniform(0.15, 0.35),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(0.05, 0.30)
        ], dtype=np.float32)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.target_gripper = float(np.random.uniform(
            self.model.jnt_range[5, 0],
            self.model.jnt_range[5, 1]
        ))

    def _set_target(self, target_pos, target_quat, target_gripper):
        if target_pos != None:
            self.target_pos = target_pos
        if target_quat != None:
            self.target_quat = target_quat
        if target_gripper != None:
            self.target_gripper = target_gripper

    def reset(self, seed=None, reposition=True, target_pos=None, target_quat=None, target_gripper=None):
        super().reset(seed=seed)

        if reposition:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:6] = np.random.uniform(-0.2, 0.2, size=6)

        if target_pos or target_quat or target_gripper:
            self._set_target()
        else:
            self._sample_target()

        try:
            target_mocap_id = self.model.body("target").mocapid[0]
            self.data.mocap_pos[target_mocap_id] = self.target_pos
            self.data.mocap_quat[target_mocap_id] = self.target_quat
        except Exception:
            pass # Skip if no target body exists in XML

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        scaled_action = np.clip(action, -1.0, 1.0) * 0.05
        current_qpos = self.data.qpos[:6]
        target_qpos = current_qpos + scaled_action

        min_limits = self.model.jnt_range[:6, 0]
        max_limits = self.model.jnt_range[:6, 1]
        self.data.ctrl[:6] = np.clip(target_qpos, min_limits, max_limits)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # ----------------------------------------------------
        # REWARD FUNCTION
        # ----------------------------------------------------
        ee_pos = self.data.body("gripper").xpos
        ee_quat = self.data.body("gripper").xquat
        current_gripper = self.data.qpos[5]

        error_pos = np.linalg.norm(ee_pos - self.target_pos)

        quat_dot = np.abs(np.dot(ee_quat, self.target_quat))
        error_quat = 1.0 - np.clip(quat_dot, 0.0, 1.0)

        error_gripper = np.abs(current_gripper - self.target_gripper)

        reward_pos = -2.0 * error_pos
        reward_quat = -4.0 * error_quat
        reward_gripper = -1.0 * error_gripper

        success = (error_pos < 0.01) and (error_quat < 0.01) and (error_gripper < 0.01)
        reward_success = 10.0 if success else 0.0
        reward = reward_pos + reward_quat + reward_gripper + reward_success

        # ----------------------------------------------------
        # TERMINATION CONDITIONS
        # ----------------------------------------------------
        terminated = bool(success)
        truncated = False

        return obs, reward, terminated, truncated, {
            "success": success,
            "error_position": error_pos,
            "error_orientation": error_quat,
            "error_gripper": error_gripper
        }