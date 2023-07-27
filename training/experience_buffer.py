import numpy as np
import torch

class ExperienceBatch():
  def __init__(self, current_states, actions, rewards, next_states, done) -> None:
    self.current_states = current_states
    self.actions = actions
    self.rewards = rewards
    self.next_states = next_states
    self.done = done

  def get_current_states(self):
    return torch.Tensor(self.current_states)

  def get_actions(self):
    return self.actions

  def get_rewards(self):
    return self.rewards

  def get_next_states(self):
    return torch.Tensor(self.next_states)

  def get_done(self):
    return self.done  

  def items(self):
    """
    ACHTUNG: not so effecient; more for testing than using in training
    """
    return list(zip(self.current_states, self.actions, self.rewards, self.next_states, self.done))


class ExperienceBuffer():
  DEFAULT_BATCH_SIZE = 128
  DEFAULT_CAPACITY = 1000

  def __init__(self, shape: tuple[int], batch_size : int = DEFAULT_BATCH_SIZE, capacity: int = DEFAULT_CAPACITY) -> None:
    """
      Parameters:
        - shape: Describes the size of each component.
          Example:
            - A 'state' is represented by 420 numbers
            - Reward is a scalar value
            - Action is a scalar value
            - Then the shape will be (420, 1, 1, 420)
        - capacity: Defines the total capacity of the buffer

      Internal Represenation:
        - We define an empty numpy arrays with value (-1) to represent empty record
          and populate them with values as the buffer gets filled
    """
    self.batch_size = batch_size
    self.capacity = capacity
    self.current_index = 0
    self.current_states = np.zeros((capacity, shape[0])) - 1
    self.actions = np.zeros((capacity, shape[1])) - 1
    self.rewards = np.zeros((capacity, shape[2])) - 1
    self.next_states = np.zeros((capacity, shape[3])) - 1
    self.done = np.zeros((capacity, shape[4])) - 1

  def add(self, current_state: list[int], action: int, reward: int, next_state: list[int], done: int) -> None:
    self.current_states[self.current_index] = current_state
    self.actions[self.current_index] = action
    self.rewards[self.current_index] = reward
    self.next_states[self.current_index] = next_state
    self.done[self.current_index] = done
    
    self.update_index()

  def update_index(self) -> None:
    """
    Updates the index that is being used to populate the buffer; 
    the 'current_index' is set to 0 whenever we reach the end of the buffer
    to override the very first values that were added (FIFO).
    """
    if self.current_index == (self.capacity - 1):
      self.current_index = 0
    else:
      self.current_index += 1

  def get_next(self) -> ExperienceBatch:
    """
    If the buffer hasn't reached the 'buffer_size' yet, then we return a batch
    representing all the data in the buffer. Otherwise we sample 'buffer_size' 
    items from the buffer.
    """
    if self.size() <= self.batch_size:
      # Case A: Current buffer size is smaller than requested => Return all items
      indices = np.arange(0, self.size())
    elif self.batch_size == 1:
      # Case B: Requested batch size is 1 => Return last item
      indices = [self.current_index - 1]
    else:
      # Case C: Requested batch size is smaller to buffer size => Sample from available items
      indices = np.random.choice(self.size(), size=self.batch_size, replace=False)
    return ExperienceBatch(*self.items(indices))

  def size(self) -> int:
    return np.count_nonzero(self.actions != -1)

  def items(self, indices=None) -> tuple:
    return (self.current_states[indices],
      self.actions[indices],
      self.rewards[indices],
      self.next_states[indices],
      self.done[indices]
    )

  def __str__(self) -> str:
    return f"""
      Capacity: {self.capacity}
      Size: {self.size()}
      Current Index: {self.current_index}
      Shape: ({self.current_states.shape[1]},{self.actions.shape[1]},{self.rewards.shape[1]},{self.next_states.shape[1]})
    """

