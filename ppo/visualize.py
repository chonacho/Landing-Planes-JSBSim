
import gymnasium as gym
import jsbgym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import gym_make
from tasks import LandingTask
gym_make.main()


# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')
env = gym.make("C172-LandingTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")

env.reset()
model=PPO.load(os.path.join("train", "model"))
env.render()
for episode in range(1, 6):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
env.close()
