import torch
import torch.nn as nn

class RewardModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.reward_model = None # to be overwritten in subclass

    def reward(self, state, action):
        return self.reward_model.reward(state, action)
