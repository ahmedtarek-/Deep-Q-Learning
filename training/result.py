import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

class Result():
  RESULTS_ROOT_DIR = "results"
  MODELS_DIR = RESULTS_ROOT_DIR + "/" + "models"

  def __init__(self, model, rewards, disc_rewards, env_name):
    self.model = model
    self.rewards = rewards
    self.disc_rewards = disc_rewards
    self.env_name = env_name

  def save(self, plot=False):
    # 1. Save model
    file_prefix = self.file_prefix()
    root = "../" + self.MODELS_DIR + "/" + self.parent_dir()

    if not os.path.exists(root):
      os.mkdir(root)

    model_path = root + "/" + self.file_prefix() + ".pt"

    torch.save(self.model.state_dict(), model_path)

    if plot:
      plt.plot(self.rewards)
      root = "../results/plots/"

      if not os.path.exists(root):
        os.mkdir(root)
      
      plt.savefig(root + file_prefix + self.env_name + ".png")

  def file_prefix(self):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H:%M")
    return dt_string + "_model"

  def parent_dir(self):
    return "1d_track" if self.env_name == "LinearTrack-v0" else "2d_track"
