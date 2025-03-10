import jsbgym
import gymnasium as gym
import numpy as np
from agent import DQNAgent
import os
import glob

# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

def find_checkpoint(episode_num):
    checkpoint_dir = "checkpoints"
    pattern = f"{checkpoint_dir}/dqn_checkpoint_ep{episode_num}_*.pth"
    matches = glob.glob(pattern)

    if not matches:
        raise ValueError(f"No checkpoint found for episode {episode_num}")
    return max(matches)

def visualize(episode_num):
    checkpoint_path = find_checkpoint(episode_num)
    print(f"Found checkpoint: {checkpoint_path}")

    env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DQNAgent(state_size, action_size)
    episode = agent.load(checkpoint_path)
    print(f"Loaded checkpoint from episode {episode}")

    state, _ = env.reset()
    total_reward = 0
    rewards = []

    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        state = next_state

        env.render()

        if terminated or truncated:
            break

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"Episode finished with average reward per step: {avg_reward:.4f}")
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trained agent')
    parser.add_argument('episode', type=int, help='Episode number to load checkpoint from')
    args = parser.parse_args()

    visualize(args.episode)