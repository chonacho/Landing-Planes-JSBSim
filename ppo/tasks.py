import gymnasium as gym
import numpy as np
import random
import types
import math
import enum
import warnings
from collections import namedtuple
import jsbgym.properties as prp
from jsbgym import assessors, rewards, utils
from jsbgym.simulation import Simulation
from jsbgym.properties import BoundedProperty, Property
from jsbgym.aircraft import Aircraft
from jsbgym.rewards import RewardStub
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type

class AltitudeTask(FlightTask):
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    DEFAULT_EPISODE_TIME_S = 60.0
    ALTITUDE_SCALING_FT = 150
    MIN_STATE_QUALITY = 0.0
    MAX_ALTITUDE_DEVIATION_FT = 1000

    altitude_error_ft = BoundedProperty(
        "error/altitude-error-ft",
        "error to desired altitude [ft]",
        prp.altitude_sl_ft.min,
        prp.altitude_sl_ft.max,
    )

    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(
        self,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        :param episode_time_s: maximum episode time in seconds
        :param positive_rewards: whether to use positive rewards (True) or negative rewards (False)
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (self.altitude_error_ft,)
        self.state_variables = (
            FlightTask.base_state_variables + self.extra_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor()
        super().__init__(assessor)

    def make_assessor(self) -> assessors.AssessorImpl:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
            ),
        )
        return assessors.AssessorImpl(
            base_components, (), positive_rewards=self.positive_rewards
        )

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
        }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        terminal_step = sim[self.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY
        return terminal_step or state_out_of_bounds or self._altitude_out_of_bounds(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        reward_scalar = (1 + sim[self.steps_left]) * -1.0
        return RewardStub(reward_scalar, reward_scalar)

    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation
    ) -> rewards.Reward:
        if self._altitude_out_of_bounds(sim) and not self.positive_rewards:
            # if using negative rewards, need to give a big negative reward on terminal
            return self._get_out_of_bounds_reward(sim)
        else:
            return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (
            prp.altitude_sl_ft,
            self.altitude_error_ft,
            prp.pitch_rad,
            prp.q_radps,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )
