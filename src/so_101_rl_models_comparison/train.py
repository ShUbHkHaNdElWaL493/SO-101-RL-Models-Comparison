from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from so_101_rl_models_comparison.so101_env import SO101Env

def make_env():
    """Utility function for multiprocessed env."""
    def initialize_env():
        env = SO101Env(xml_path="models/scene.xml")
        env = TimeLimit(env, max_episode_steps=300)
        env = Monitor(env)
        return env
    return initialize_env

def main():
    model_type = "SAC"    
    vec_env = SubprocVecEnv([make_env() for _ in range(8)])

    if model_type == "PPO":
        # ---------------------------------------------------------
        # Option A: Train with PPO (Proximal Policy Optimization)
        # ---------------------------------------------------------
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            tensorboard_log=f"./log/{model_type}",
            device="cuda"
        )

    elif model_type == "SAC":
        # ---------------------------------------------------------
        # Option B: Train with SAC (Soft Actor-Critic)
        # ---------------------------------------------------------
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=800_000,
            learning_starts=8000,
            batch_size=256,
            tensorboard_log=f"./log/{model_type}",
            device="cuda"
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=80_000,
        save_path=f"./log/{model_type}/checkpoints",
        name_prefix=f"so101_{model_type}_"
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    model.save(f"so101_{model_type}")
    print(f"so101_{model_type} model saved.")

if __name__ == '__main__':
    main()