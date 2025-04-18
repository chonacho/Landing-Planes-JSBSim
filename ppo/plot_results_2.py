import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gym_make
from tasks import AltitudeTask
gym_make.main()
# Custom gym environment
ENV_NAME = "C172-CustomTurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0"

# Path to the folder containing the models
MODEL_DIR = "train"

# Model naming pattern
MODEL_PREFIX = "best_model_"

# Initialize lists to store statistics
model_numbers = []
min_rewards = []
max_rewards = []
avg_rewards = []

# Iterate over each model file in the directory
for model_num in range(100000, 10000001, 100000):
    model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{model_num}")
    print(model_path)
    # Load the model
    model = PPO.load(model_path)

    # Evaluate the model across 10 different seeds
    rewards = []
    env = gym.make(ENV_NAME)
    for seed in range(10):
        # env.seed(seed)
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            for i in range(20):
                action1, _ = model.predict(obs)
                action += action1
            action /= 21
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)

    # Calculate statistics
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    avg_reward = np.mean(rewards)

    # Store the statistics
    model_numbers.append(model_num)
    min_rewards.append(min_reward)
    max_rewards.append(max_reward)
    avg_rewards.append(avg_reward)

plt.figure(figsize=(10, 6))
plt.plot(model_numbers, min_rewards, label='Min Reward')
plt.plot(model_numbers, max_rewards, label='Max Reward')
plt.plot(model_numbers, avg_rewards, label='Avg Reward')
plt.xlabel('Model Number')
plt.ylabel('Reward')
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
plt.legend()
plt.title('Evaluation of Models')
plt.grid()
plt.show()