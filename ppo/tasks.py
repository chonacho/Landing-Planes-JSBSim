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

from jsbgym.tasks import Task, FlightTask, Shaping

class LandingTask(FlightTask):
    """
    Attempting to land
    """

    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 90.0
    ALTITUDE_SCALING_FT = 150
    TRACK_ERROR_SCALING_DEG = 8
    ROLL_ERROR_SCALING_RAD = 0.15  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.0
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 3000  # terminate if altitude error exceeds this
    target_track_deg = BoundedProperty(
        "target/track-deg",
        "desired heading [deg]",
        prp.heading_deg.min,
        prp.heading_deg.max,
    )
    track_error_deg = BoundedProperty(
        "error/track-error-deg", "error to desired track [deg]", -180, 180
    )
    altitude_error_ft = BoundedProperty(
        "error/altitude-error-ft",
        "error to desired altitude [ft]",
        -MAX_ALTITUDE_DEVIATION_FT,
        MAX_ALTITUDE_DEVIATION_FT,
    )
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd)

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.step_frequency_hz = 0.3 #seeing if this makes the agent do less jerky movements.
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (
            self.altitude_error_ft,
            self.track_error_deg,
            prp.sideslip_deg,
        )
        self.state_variables = (
            FlightTask.base_state_variables + self.extra_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
            ),
            rewards.AsymptoticErrorComponent(
                name="travel_direction",
                prop=self.track_error_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.TRACK_ERROR_SCALING_DEG,
            ),
            # add an airspeed error relative to cruise speed component?
        )
        return base_components

    def _select_assessor(
        self,
        base_components: Tuple[rewards.RewardComponent, ...],
        shaping_components: Tuple[rewards.RewardComponent, ...],
        shaping: Shaping,
    ) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            wings_level = rewards.AsymptoticErrorComponent(
                name="wings_level",
                prop=prp.roll_rad,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.ROLL_ERROR_SCALING_RAD,
            )
            no_sideslip = rewards.AsymptoticErrorComponent(
                name="no_sideslip",
                prop=prp.sideslip_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG,
            )
            potential_based_components = (wings_level, no_sideslip)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(
                base_components,
                potential_based_components,
                positive_rewards=self.positive_rewards,
            )
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            altitude_error, travel_direction = base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {wings_level: (travel_direction,)}
            return assessors.ContinuousSequentialAssessor(
                base_components,
                potential_based_components,
                potential_dependency_map=dependency_map,
                positive_rewards=self.positive_rewards,
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
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
        }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_track_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_track_error(self, sim: Simulation):
        v_north_fps, v_east_fps = sim[prp.v_north_fps], sim[prp.v_east_fps]
        track_deg = prp.Vector2(v_east_fps, v_north_fps).heading_deg()
        target_track_deg = sim[self.target_track_deg]
        error_deg = utils.reduce_reflex_angle_deg(track_deg - target_track_deg)
        sim[self.track_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        # TODO: issues if sequential?
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY
        return terminal_step or state_out_of_bounds or self._altitude_out_of_bounds(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[self.steps_left]) * -2.0
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
        sim[self.target_track_deg] = self._get_target_track()

    def _get_target_track(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            self.altitude_error_ft,
            self.track_error_deg,
            prp.roll_rad,
            prp.sideslip_deg,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )

class AltitudeTask(FlightTask):
    """
    A task in which the agent must maintain a constant altitude while keeping level flight.
    """

    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    DEFAULT_EPISODE_TIME_S = 60.0
    ALTITUDE_SCALING_FT = 150
    ROLL_ERROR_SCALING_RAD = 0.05  # approx. 8 deg
    PITCH_ERROR_SCALING_RAD = 0.05  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.0
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 1000  # terminate if altitude error exceeds this
    MAX_ATTITUDE_DEG = 30  # terminate if roll or pitch exceeds this

    altitude_error_ft = BoundedProperty(
        "error/altitude-error-ft",
        "error to desired altitude [ft]",
        prp.altitude_sl_ft.min,
        prp.altitude_sl_ft.max,
    )
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd)

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.step_frequency_hz = 0.3
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (self.altitude_error_ft, prp.sideslip_deg)
        self.state_variables = (
            FlightTask.base_state_variables + self.extra_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
            ),
            rewards.AsymptoticErrorComponent(
                name="roll_error",
                prop=prp.roll_rad,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ROLL_ERROR_SCALING_RAD,
            ),
        )
        return base_components

    def _select_assessor(
        self,
        base_components: Tuple[rewards.RewardComponent, ...],
        shaping_components: Tuple[rewards.RewardComponent, ...],
        shaping: Shaping,
    ) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            wings_level = rewards.AsymptoticErrorComponent(
                name="wings_level",
                prop=prp.roll_rad,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.ROLL_ERROR_SCALING_RAD,
            )
            no_sideslip = rewards.AsymptoticErrorComponent(
                name="no_sideslip",
                prop=prp.sideslip_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG,
            )
            pitch_error = rewards.AsymptoticErrorComponent(
                name="pitch_error",
                prop=prp.pitch_rad,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.PITCH_ERROR_SCALING_RAD,
            )
            potential_based_components = (wings_level, no_sideslip, pitch_error)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(
                base_components,
                potential_based_components,
                positive_rewards=self.positive_rewards,
            )
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            altitude_error, roll_error= base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {no_sideslip: (altitude_error,), pitch_error: (altitude_error,)}
            return assessors.ContinuousSequentialAssessor(
                base_components,
                potential_based_components,
                potential_dependency_map=dependency_map,
                positive_rewards=self.positive_rewards,
            )
            # No additional shaping rewards needed since we already penalize attitude deviations
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
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
        return (
            terminal_step
            or state_out_of_bounds
            or self._altitude_out_of_bounds(sim)
            #or self._attitude_out_of_bounds(sim)
        )

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _attitude_out_of_bounds(self, sim: Simulation) -> bool:
        roll_deg = math.degrees(sim[prp.roll_rad])
        pitch_deg = math.degrees(sim[prp.pitch_rad])
        return abs(roll_deg) > self.MAX_ATTITUDE_DEG or abs(pitch_deg) > self.MAX_ATTITUDE_DEG

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[self.steps_left]) * -1.0
        return RewardStub(reward_scalar, reward_scalar)

    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation
    ) -> rewards.Reward:
        if (self._altitude_out_of_bounds(sim) or self._attitude_out_of_bounds(sim)) and not self.positive_rewards:
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
            prp.u_fps,
            prp.altitude_sl_ft,
            self.altitude_error_ft,
            prp.roll_rad,
            prp.pitch_rad,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )

class CustomHeadingControlTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    initial heading.
    """

    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    INITIAL_ALTITUDE_FT=5000
    DEFAULT_EPISODE_TIME_S = 60.0
    ALTITUDE_SCALING_FT = 100
    TRACK_ERROR_SCALING_DEG = 8
    ROLL_ERROR_SCALING_RAD = 0.15  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.0
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 3000  # terminate if altitude error exceeds this
    target_track_deg = BoundedProperty(
        "target/track-deg",
        "desired heading [deg]",
        prp.heading_deg.min,
        prp.heading_deg.max,
    )
    target_altitude_ft = BoundedProperty(
        "target/altitude-ft",
        "desired altitude [ft]",
        INITIAL_ALTITUDE_FT-1000,
        INITIAL_ALTITUDE_FT+1000,
    )

    track_error_deg = BoundedProperty(
        "error/track-error-deg", "error to desired track [deg]", -180, 180
    )
    altitude_error_ft = BoundedProperty(
        "error/altitude-error-ft",
        "error to desired altitude [ft]",
        -MAX_ALTITUDE_DEVIATION_FT,
        MAX_ALTITUDE_DEVIATION_FT,
    )
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (
            self.altitude_error_ft,
            self.track_error_deg,
            prp.sideslip_deg,
        )
        self.state_variables = (
            FlightTask.base_state_variables + self.extra_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
            ),
            rewards.AsymptoticErrorComponent(
                name="travel_direction",
                prop=self.track_error_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.TRACK_ERROR_SCALING_DEG,
            ),
            # add an airspeed error relative to cruise speed component?
        )
        return base_components

    def _select_assessor(
        self,
        base_components: Tuple[rewards.RewardComponent, ...],
        shaping_components: Tuple[rewards.RewardComponent, ...],
        shaping: Shaping,
    ) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            wings_level = rewards.AsymptoticErrorComponent(
                name="wings_level",
                prop=prp.roll_rad,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.ROLL_ERROR_SCALING_RAD,
            )
            no_sideslip = rewards.AsymptoticErrorComponent(
                name="no_sideslip",
                prop=prp.sideslip_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=True,
                scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG,
            )
            potential_based_components = (wings_level, no_sideslip)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(
                base_components,
                potential_based_components,
                positive_rewards=self.positive_rewards,
            )
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            altitude_error, travel_direction = base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {wings_level: (travel_direction,), no_sideslip: (altitude_error,)}
            return assessors.ContinuousSequentialAssessor(
                base_components,
                potential_based_components,
                potential_dependency_map=dependency_map,
                positive_rewards=self.positive_rewards,
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
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
        }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_track_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_track_error(self, sim: Simulation):
        v_north_fps, v_east_fps = sim[prp.v_north_fps], sim[prp.v_east_fps]
        track_deg = prp.Vector2(v_east_fps, v_north_fps).heading_deg()
        target_track_deg = sim[self.target_track_deg]
        error_deg = utils.reduce_reflex_angle_deg(track_deg - target_track_deg)
        sim[self.track_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = sim[self.target_altitude_ft]
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        # TODO: issues if sequential?
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY
        return terminal_step or state_out_of_bounds or self._altitude_out_of_bounds(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
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
        sim[self.target_track_deg] = self._get_target_track()
        sim[self.target_altitude_ft] = self._get_target_altitude()

    def _get_target_track(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            self.target_altitude_ft,
            self.altitude_error_ft,
            self.target_track_deg,
            self.track_error_deg,
            prp.roll_rad,
            prp.sideslip_deg,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )


class CustomTurnHeadingControlTask(CustomHeadingControlTask):
    """
    A task in which the agent must make a turn from a random initial heading,
    and fly level to a random target heading.
    """

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT+ random.uniform(-1000, 1000)

    def _get_target_track(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_track_deg.min, self.target_track_deg.max)