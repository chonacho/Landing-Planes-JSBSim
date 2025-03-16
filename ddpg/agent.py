import gymnasium as gym
import jsbgym
import importlib
import os
import torch as th
from torch import nn
import numpy as np
from jsbgym.tests.stubs import FlightTaskStub
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym_make
from tasks import AltitudeTask
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


env = gym.make("C172-AltitudeTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0")

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

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
policy_kwargs = dict(
    #features_extractor_class=CustomNetwork,
    #features_extractor_kwargs=dict(features_dim=256),
    #net_arch=dict(pi=[256, 256], vf=[256, 256]),
    log_std_init=-5,
    n_critics=3,
)

model = SAC(
    "MlpPolicy", 
    env,
    action_noise=action_noise, 
    gamma=0.99,
    batch_size=100,
    use_sde_at_warmup=True,
    train_freq=10,
    policy_kwargs=policy_kwargs,
    learning_starts=100000,
    tau=0.005,
    verbose=1)
model.set_env(env, force_reset=True)
model.learn(total_timesteps=1500000, callback=callback, progress_bar=True, log_interval=10000)
model.save("model")

print("done")
