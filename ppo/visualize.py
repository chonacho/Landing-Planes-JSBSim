import gymnasium as gym
import jsbgym
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np
import torch
import gym_make
from tasks import AltitudeTask
gym_make.main()
from stable_baselines3.common.policies import obs_as_tensor

def predict_proba(model, state, lstm_states, episode_starts):
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs, lstm_states, episode_starts)
    #print(dis)
    probs = dis.distribution.mean
    std_np = dis.distribution.stddev.detach().numpy()
    probs_np = probs.detach().numpy()

    return np.divide(probs_np, std_np)

# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

env = gym.make("C172-CustomTurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-FG-v0", render_mode="flightgear")

env.reset()
model = RecurrentPPO.load("model")
#model = PPO.load(os.path.join("train", "best_model_200000"))
env.render()

#for episode in range(1, 6):
episode = 0
while True:
    obs, _ = env.reset()
    done = False
    total_reward = 0
    lstm_states = None
    while not done:
        action, lstm_states = model.predict(obs, lstm_states)
        #print(action)
        #for i in range(30):
        #    action1, _ = model.predict(obs)
        #    action+= action1
        #action /=31
        # action = np.clip(predict_proba(model, obs, lstm_states, episode_starts)[0], min=-1, max=1)
        #print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        #print(obs[0])
        #print(model.action_probability(obs))
        done = terminated or truncated
        total_reward += reward
    print(f"Total Reward for episode {episode} is {total_reward}")
    episode+=1
env.close()
