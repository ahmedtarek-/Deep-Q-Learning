"""
Main entry point to run training
"""
# 1. Imports
import sys
sys.path.append('../')

import gym
import gym_grid

import numpy as np
import torch
import torch.nn as nn
import yaml

from trainer import Trainer
from result import Result

# 2. Load configuration
with open('config.yaml') as f:
  config = yaml.safe_load(f)['training_config']

# 3. Load and initialize trainer object (all the config goes in here)
trainer = Trainer(env_name=config["env_name"])

# 4. Run a training loop
training_rewards, training_discounted_rewards = trainer.the_loop()

# 5. Save the model and plot
Result(
  model=trainer.model,
  rewards=training_rewards,
  disc_rewards=training_discounted_rewards, 
  env_name=config["env_name"]).save(plot=True)
