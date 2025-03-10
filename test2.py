import jsbgym
import gymnasium as gym
import numpy as np

# Weird necessary fix
from jsbgym import visualiser
import logging
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
observation, info = env.reset()

terminated = False
while not terminated:
    observation, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
    env.render()