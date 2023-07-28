import torch
import numpy as np

def eps_greedy(q, e):
  rand = np.random.uniform(0, 1)
  if len(q.shape) == 1:
    q = torch.unsqueeze(q, dim=0)

  if rand <= (1 - e):
    chosen_action = torch.argmax(q, dim=1)
    q_value = q[np.arange(len(q)), chosen_action]
  else:
    chosen_action = np.random.choice(range(q.shape[1]))
    q_value = q[:,chosen_action]

  return chosen_action, q_value
