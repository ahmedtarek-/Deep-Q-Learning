import gym
import sys
import numpy as np
import matplotlib.pyplot as plt

from pycolab import ascii_art, rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


MAZES_ART = [
    ['##############',
     '#P $ $a$ $ $ #',
     '# @ $   $ @ $#',
     '#$ @ $ $ $ @ #',
     '# $ $   $ $ $#',
     '##############']]


# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '$': (999, 862, 110),  # Shimmering golden coins
             '*': (185, 242, 255),  # Diamond
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # Player
             'a': (999, 0, 780),    # Patroller A
             '@': (145, 987, 341)}  # Poison


def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  return ascii_art.ascii_art_to_game(
      MAZES_ART[level], what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': PatrollerSprite},
      drapes={
          '$': CashDrape,
          '@': PoisonDrape},
      update_schedule=['P', '$', '@', 'a'],
      z_order='@$aP')


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')
    self.num_steps = 0

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused
    self.num_steps += 1

    if actions == 0:    # go upward
      self._north(board, the_plot)
    elif actions == 1:  # go downward
      self._south(board, the_plot)
    elif actions == 2:  # go leftward
      self._west(board, the_plot)
    elif actions == 3:  # go rightward
      self._east(board, the_plot)
    elif actions == 4:  # stay put
      self._stay(board, the_plot)
    if self.num_steps == 100: # terminate when reached max episode steps
      the_plot.terminate_episode()
      self.num_steps = 0


class CashDrape(plab_things.Drape):
  """A `Drape` handling all of the coins.

  This Drape detects when a player traverses a coin, removing the coin and
  crediting the player for the collection. Terminates if all coins are gone.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached a coin, credit reward 100 and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    player_pattern_position = things['P'].position

    if self.curtain[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(100)
      self.curtain[player_pattern_position] = False
      if not self.curtain.any(): the_plot.terminate_episode()

class PoisonDrape(plab_things.Drape):
  """A `Drape` handling all of the poisons.

  This Drape detects when a player traverses a poison and
  punishing the player for the stepping on.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has stepped  on a poison, credit reward -100.
    player_pattern_position = things['P'].position

    if self.curtain[player_pattern_position]:
      the_plot.log('Poison absorbed at {}!'.format(player_pattern_position))
      the_plot.add_reward(-100)
      self.curtain[player_pattern_position] = False    

class PatrollerSprite(prefab_sprites.MazeWalker):
  """Wanders back and forth horizontally, killing the player on contact."""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(PatrollerSprite, self).__init__(
        corner, position, character, impassable='#')
    # Choose our initial direction based on our character value.
    self._moving_south = bool(ord(character) % 2)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)  # Also not strictly necessary.
      return

    # If there is a wall next to us, we ought to switch direction.
    row, col = self.position
    if layers['#'][row - 1, col]: self._moving_south = True
    if layers['#'][row + 1, col]: self._moving_south = False

    # Make our move. If we're now in the same cell as the player, it's instant
    # game over!
    (self._south if self._moving_south else self._north)(board, the_plot)
    if self.position == things['P'].position: the_plot.terminate_episode()

class DeadlyGridEnv(gym.Env):
    """
    Wrapper to adapt to OpenAI's gym interface.
    """
    action_space = gym.spaces.Discrete(5)
    observation_space = gym.spaces.Box(low=0, high=1, shape=[6, 14, 5], dtype=np.uint8)

    def _to_obs(self, observation):
        hallway = observation.layers[' ']
        ob = np.stack([observation.layers[c] for c in '@$aP'] + [hallway], axis=2).astype(np.uint8)
        return ob

    def reset(self):
        self._game = make_game(0)
        observation, _, _ = self._game.its_showtime()
        return self._to_obs(observation)

    def reset_with_render(self):
        self._game = make_game(0)
        observation, _ , _ = self._game.its_showtime()
        return self._to_obs(observation), observation

    def step(self, action):
        observation, reward, _ = self._game.play(action)
        if reward is None: reward = 0
        done = self._game.game_over
        info = {}
        return self._to_obs(observation), reward, done, info

    def step_with_render(self, action):
        observation, reward, _ = self._game.play(action)
        if reward is None: reward = 0
        done = self._game.game_over
        info = {}
        return self._to_obs(observation), reward, done, info, observation

    def render(self, obs_to_render):
        converted = converter(obs_to_render)
        plt.imshow(converted)
        plt.axis("off")
        return

"""
Helper functions/objects to render the environment
- RGB rescaler to 0,1 range
- Color scheme for relevant environment objects
- Transform obervation into "imshow"-able array
"""

def rgb_rescale(v):
    return v/255


COLOUR_FG = {' ': tuple([rgb_rescale(v) for v in (255, 255, 255)]), # Background
             '$': tuple([rgb_rescale(v) for v in (214, 182, 79)]),  # Coins
             '@': tuple([rgb_rescale(v) for v in (66, 6, 13)]),     # Poison
             '#': tuple([rgb_rescale(v) for v in (119, 107, 122)]), # Walls of the maze
             'P': tuple([rgb_rescale(v) for v in (153, 85, 74)]),   # Player
             'a': tuple([rgb_rescale(v) for v in (107, 132, 102)]), # Patroller A
             'b': tuple([rgb_rescale(v) for v in (107, 132, 102)]), # Patroller B
             '*': tuple([rgb_rescale(v) for v in(185, 242, 255)]),  # Diamond
             } 


def converter(obs):
    converter = rendering.ObservationToArray(COLOUR_FG, permute=(0,1,2))
    converted = np.swapaxes(converter(obs), 1, 2).T
    return converted
