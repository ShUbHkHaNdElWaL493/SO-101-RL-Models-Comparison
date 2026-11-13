import argparse
from gymnasium.wrappers import TimeLimit
import mujoco.viewer
from stable_baselines3 import PPO, SAC
import time

from so_101_rl_models_comparison.so101_env import SO101Env

def test(mjcf_file, model_type='PPO', episodes=5):
    env = SO101Env(mjcf_file=mjcf_file)
    env = TimeLimit(env, max_episode_steps=300)

    model_file = f'so101_{model_type}.zip'
    print(f'Loading {model_type} model from {model_file}...')
    if model_type == 'PPO':
        model = PPO.load(model_file)
    elif model_type == 'SAC':
        model = SAC.load(model_file)
    else:
        raise ValueError('Unsupported model type.')

    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
        for episode in range(episodes):
            obs, info = env.reset()
            viewer.sync()

            terminated = False
            truncated = False
            episode_reward = 0
            
            print(f'\n--- Episode {episode + 1} ---')
            
            while not (terminated or truncated):
                if not viewer.is_running():
                    print('Viewer closed by user. Exiting evaluation.')
                    return

                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                viewer.sync()
                time.sleep(0.02)

            print(f'Success: {info.get('success', False)}')
            print(f'Reward: {episode_reward:.2f}')
            print(f'Position Error: {info.get('error_position', 0.0)}')
            print(f'Orientation Error: {info.get('error_orientation', 0.0)}')
            print(f'Gripper Error: {info.get('error_gripper', 0.0)}')

    env.close()

def main():
    parser = argparse.ArgumentParser(description='Test an RL model for the LeRobot SO-101 robot arm environment.')
    parser.add_argument('-f', '--mjcf_file', type=str, default='models/so101_scene.xml', help='Path to the MJCF file.')
    parser.add_argument('-m', '--model_type', type=str, default='SAC', help='Type of RL model to test.')
    parser.add_argument('-e', '--episodes', type=int, default=10, help='Number of episodes for testing the model.')
    args = parser.parse_args()
    test(mjcf_file=args.mjcf_file, model_type=args.model_type, episodes=args.episodes)

if __name__ == '__main__':
    main()