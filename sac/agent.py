import gymnasium as gym
import numpy as np
import jsbgym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from stable_baselines3.sac.policies import LnMlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


env = gym.make("C172-TurnHeadingControlTask-Shaping.STANDARD-NoFG-v0", render_mode="human")
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

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
#model.learn(total_timesteps=2000, callback=callback, progress_bar=True, log_interval=10)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = SAC("MlpPolicy", env, verbose=1,action_noise=action_noise, gamma=0.99, learning_rate=0.001, learning_starts=5000, target_update_interval=10)
model.learn(total_timesteps=20000, log_interval=10, progress_bar=True, callback=callback)
model.save("JSBSim_20000_steps")

print("done")
