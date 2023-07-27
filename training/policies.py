import torch
import numpy as np

def eps_greedy(q, e):
  rand = np.random.uniform(0, 1)
  if len(q.shape) == 1:
    q = torch.unsqueeze(q, dim=0)

  #print(q)
  if rand <= (1 - e):
    #chosen_action = torch.argmax(q, dim=1).item()
    chosen_action = torch.argmax(q, dim=1)
  else:
    chosen_action = np.random.choice(range(q.shape[1]))

  # to avod the warning
  #print(f"Q dim is {q.shape}")
  q_value = q[:,chosen_action]
  #print(f"Just q value: {q_value}")
  return chosen_action, q_value
