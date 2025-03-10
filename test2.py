import jsbgym
import gymnasium as gym
import os
import logging
import sys
from jsbgym import visualiser
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('jsbgym')

# Patch the visualiser module's logger
visualiser.gym.logger = logger

# Set FlightGear environment variables
os.environ['FG_ROOT'] = 'C:/Users/notan/FlightGear/Downloads/fgdata_2024_1'
os.environ['FG_AIRCRAFT'] = os.path.join(os.environ['FG_ROOT'], 'Aircraft')

# Print debug information
logger.info(f"FG_ROOT set to: {os.environ['FG_ROOT']}")
logger.info(f"FG_AIRCRAFT set to: {os.environ['FG_AIRCRAFT']}")

# Create and initialize the environment
env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
observation, info = env.reset()

try:
    logger.info("Attempting to render environment...")
    env.render()
    logger.info("Render successful!")

    # Simple simulation loop
    logger.info("Starting simulation loop...")
    for i in range(100):  # Run for 100 steps
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)  # Small delay to make visualization smoother

        if terminated or truncated:
            logger.info("Episode ended, resetting environment...")
            observation, info = env.reset()

finally:
    logger.info("Closing environment...")
    env.close()
    logger.info("Environment closed.")