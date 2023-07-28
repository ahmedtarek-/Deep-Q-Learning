import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LinearTrack-v0',
    entry_point='gym_grid.envs:LinearTrackEnv',
)

# TODO:
register(
    id='DeadlyGrid-v0',
    entry_point='gym_grid.envs:DeadlyGridEnv',
)
