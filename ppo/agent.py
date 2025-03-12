import gymnasium as gym
import jsbgym
import importlib
import os
from jsbgym.tests.stubs import FlightTaskStub
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
#import gym_make
#from tasks import LandingTask
#gym_make.main()

env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-NoFG-v0")
env.reset()

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
            print()
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, ent_coef=0.1, learning_rate=0.0001, gamma=0.999, verbose=1)
model.learn(total_timesteps=1000000, callback=callback, progress_bar=True, log_interval=10000)
model.save("model")

print("done")
