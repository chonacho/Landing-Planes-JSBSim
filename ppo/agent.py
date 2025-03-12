import gymnasium as gym
import jsbgym
import importlib
import os
import torch as th
from torch import nn
import numpy as np
from jsbgym.tests.stubs import FlightTaskStub
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tasks import *
import gym_make
gym_make.main()

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)


env = gym.make("C172-AltitudeTask-NoFG-v0")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

CHECKPOINT_DIR = "train/"
LOG_DIR = "logs/"

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)
        return True

policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.ReLU,
)

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-2,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.05,
    ent_coef=0.01,
    verbose=1
)

model.learn(total_timesteps=500000, callback=callback, progress_bar=True, log_interval=100000)
model.save("model")

print("done")
