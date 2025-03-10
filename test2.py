import jsbgym
import gymnasium as gym
import os
from jsbgym import visualiser
import logging

# Weird necessary fix
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
observation, info = env.reset()
env.render()