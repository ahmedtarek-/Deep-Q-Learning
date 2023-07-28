import torch
import matplotlib.pyplot as plt
from datetime import datetime

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
    model_path = self.MODELS_DIR + "/" + self.parent_dir() + "/" + self.file_prefix() + ".pt"

    torch.save(self.model.state_dict(), model_path)

    if plot:
      plt.plot(self.rewards)
      plt.savefig(self.RESULTS_ROOT_DIR + "/" + "plots" + "/" + self.file_prefix() + ".png")

  def file_prefix(self):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H:%M")
    return dt_string + "_model"

  def parent_dir(self):
    return "1d_track" if self.env_name == "LinearTrack-v0" else "2d_track"
