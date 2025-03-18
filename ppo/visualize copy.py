import gymnasium as gym
import jsbgym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import HumanRendering
import gym_make
from tasks import AltitudeTask
gym_make.main()

# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

env = gym.make("C172-CustomTurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0", render_mode="human")

env.reset()
#model = PPO.load("model")
model = PPO.load(os.path.join("train", "best_model_1000000"))
env.render()

#for episode in range(1, 6):
episode = 0
for ii in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        for i in range(0):
            action1, _ = model.predict(obs)
            action+= action1
        action /=1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Total Reward for episode {episode} is {total_reward}")
    episode+=1
env.close()
