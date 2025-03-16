import gymnasium as gym
import jsbgym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import gym_make
from tasks import AltitudeTask
gym_make.main()

# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

env = gym.make("C172-AltitudeTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0", render_mode="human")

env.reset()
#model = TD3.load("model")
model = SAC.load(os.path.join("train", "best_model_500000"))
env.render()

for episode in range(1, 6):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Total Reward for episode {episode} is {total_reward}")
env.close()
