import jsbgym
import gymnasium as gym
import os
from jsbgym import visualiser
import logging

# Configure logging and patch the visualiser's logger
logging.basicConfig(level=logging.INFO)
visualiser.gym.logger = logging.getLogger('jsbgym')

# Set FlightGear environment variables
os.environ['FG_ROOT'] = 'C:/Users/notan/FlightGear/Downloads/fgdata_2024_1'

# Create and initialize the environment
env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
observation, info = env.reset()
env.render()