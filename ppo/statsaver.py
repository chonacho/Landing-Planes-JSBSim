import gymnasium as gym
import jsbgym
import importlib
import os
import torch as th
from typing import Callable
from torch import nn
import numpy as np
from jsbgym.tests.stubs import FlightTaskStub
from stable_baselines3 import PPO
#from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import gym_make
from tasks import AltitudeTask, CustomHeadingControlTask, CustomTurnHeadingControlTask
gym_make.main()

MODEL_DIR = "train"

# Model naming pattern
MODEL_PREFIX = "best_model_"



ENV_NAME = "C172-CustomTurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0"

CHECKPOINT_DIR = "train/"
NORM_DIR = "normstats/"
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
            vec_env = self.model.get_env()
            vec_env.save(os.path.join(NORM_DIR, "model_{}.pkl".format(self.n_calls)))
            mean_reward, std_reward = evaluate_policy(self.model, vec_env, deterministic=True, n_eval_episodes=5, warn=False)
            print(f"{mean_reward} std: {std_reward} ")
        return True

policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    log_std_init=-2,
    ortho_init=False,
    activation_fn=nn.ReLU
)



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

env = gym.make("C172-CustomTurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0")
env = Monitor(env)
#env = VecFrameStack(env,8)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, gamma=0.99)

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
for model_num in range(100000, 10000001, 100000):
    model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{model_num}")
    stat_path = os.path.join("normstats", f"model_{model_num}.pkl")
    print(model_path)
    # Load the model


    # Evaluate the model across 10 different seeds
    rewards = []
    env = gym.make(ENV_NAME)
    env = Monitor(env, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    try:
        env = VecNormalize.load(stat_path, env)
    except:
        env = VecNormalize(env, gamma=0.99, training=True, norm_reward=False)
    """
    model = PPO(
    "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        #tensorboard_log=LOG_DIR,
        learning_rate=3e-5,
        gamma=0.99,
        gae_lambda=0.9,
        batch_size=256,
        max_grad_norm=2,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.4,
        verbose=1
    )"""
    model = PPO.load(model_path, env)
    env.reset()
    model.learn(total_timesteps=10000, progress_bar=True, log_interval=100000)
    rewards, lens = evaluate_policy(model, env, deterministic=True, n_eval_episodes=5, warn=False)
    print(rewards)
    env.save(os.path.join(NORM_DIR, "model_{}.pkl".format(model_num)))

#model.save("model")

print("done")

