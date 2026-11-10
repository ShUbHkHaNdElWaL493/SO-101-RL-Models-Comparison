from gymnasium.wrappers import TimeLimit
import mujoco.viewer
from stable_baselines3 import PPO, SAC
import time

from so_101_rl_models_comparison.so101_env import SO101Env

def evaluate(model_path, model_type="PPO", num_episodes=5):
    env = SO101Env(xml_path="models/scene.xml")
    env = TimeLimit(env, max_episode_steps=300)

    print(f"Loading {model_type} model from {model_path}...")
    if model_type == "PPO":
        model = PPO.load(model_path)
    elif model_type == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError("Unsupported model type.")

    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
        for episode in range(num_episodes):
            obs, info = env.reset()
            viewer.sync()

            terminated = False
            truncated = False
            episode_reward = 0
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while not (terminated or truncated):
                if not viewer.is_running():
                    print("Viewer closed by user. Exiting evaluation.")
                    return

                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                # Optional: If you have a mocap body named "target" in your XML, 
                # you can update its position here so you can see the target in the viewer!
                # try:
                #     target_body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "target")
                #     env.unwrapped.data.mocap_pos[target_body_id] = env.unwrapped.target_pos
                # except Exception:
                #     pass # Skip if no target body exists in XML

                viewer.sync()
                time.sleep(0.02)

            print(f"Success: {info.get('success', False)}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Position Error: {info.get('error_position', 0.0)}")
            print(f"Orientation Error: {info.get('error_orientation', 0.0)}")
            print(f"Gripper Error: {info.get('error_gripper', 0.0)}")

    env.close()

def main():
    model_type = "SAC"
    evaluate(f"so101_{model_type}", model_type=model_type, num_episodes=5)

if __name__ == '__main__':
    main()