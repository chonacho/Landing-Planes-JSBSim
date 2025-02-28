import jsbgym
import gymnasium as gym
import numpy as np
import gym_make
from jsbgym.tests.stubs import FlightTaskStub
from jsbgym.agents.agents import RandomAgent

from tasks import LandingTask
#import gym_jsbsim

#from gymnasium import envs
#all_envs = envs.registry.keys()
#print(all_envs)
#env_ids = [env_spec.id for env_spec in all_envs]
#print(sorted(env_ids))
env = gym.make("C172-LandingTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
env.reset()
#env.render()
terminated = False
action_space = env.task.get_action_space()
agent = RandomAgent(action_space=action_space)
while not terminated:
    action = agent.act(None)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(observation)
