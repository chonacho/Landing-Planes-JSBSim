import gymnasium as gym
import enum
from jsbgym.tasks import Task, HeadingControlTask, TurnHeadingControlTask
from tasks import *
from jsbgym.aircraft import Aircraft, c172



def get_env_id(aircraft, task_type, shaping, enable_flightgear) -> str:
    """
    Creates an env ID from the environment's components
    :param task_type: Task class, the environment's task
    :param aircraft: Aircraft namedtuple, the aircraft to be flown
    :param shaping: HeadingControlTask.Shaping enum, the reward shaping setting
    :param enable_flightgear: True if FlightGear simulator is enabled for visualisation else False
    """
    if enable_flightgear:
        fg_setting = "FG"
    else:
        fg_setting = "NoFG"
    return f"{aircraft.name}-{task_type.__name__}-{shaping}-{fg_setting}-v0"
def get_env_id_kwargs_map() -> Dict[str, Tuple]:
    """Returns all environment IDs mapped to tuple of (task, aircraft, shaping, flightgear)"""
    # lazy import to avoid circular dependencies
    from jsbgym.tasks import Shaping, HeadingControlTask, TurnHeadingControlTask
    from tasks import LandingTask, AltitudeTask

    map = {}
    for task_type in [LandingTask, AltitudeTask]:
        for plane in [c172]:
            for shaping in (Shaping.STANDARD, Shaping.EXTRA, Shaping.EXTRA_SEQUENTIAL):
                for enable_flightgear in (True, False):
                    id = get_env_id(plane, task_type, shaping, enable_flightgear)
                    assert id not in map
                    map[id] = (plane, task_type, shaping, enable_flightgear)
    return map

class AttributeFormatter(object):
    """
    Replaces characters that would be illegal in an attribute name

    Used through its static method, translate()
    """

    ILLEGAL_CHARS = r"\-/."
    TRANSLATE_TO = "_" * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)
def main():
    for env_id, (
        plane,
        task,
        shaping,
        enable_flightgear,
    ) in get_env_id_kwargs_map().items():
        if enable_flightgear:
            entry_point = "jsbgym.environment:JsbSimEnv"
        else:
            entry_point = "jsbgym.environment:NoFGJsbSimEnv"
        kwargs = dict(aircraft=plane, task_type=task, shaping=shaping)
        gym.envs.registration.register(id=env_id, entry_point=entry_point, kwargs=kwargs)

    # make an Enum storing every Gym-JSBSim environment ID for convenience and value safety
    Envs = enum.Enum.__call__(
        "Envs",
        [
            (AttributeFormatter.translate(env_id), env_id)
            for env_id in get_env_id_kwargs_map().keys()
        ],
    )
    print(get_env_id_kwargs_map().keys())
if __name__ == '__main__':
    main()
