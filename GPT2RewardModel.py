from models.RewardModel import RewardModel
from models.GPT2FeedbackScorer import GPT2FeedbackScorer

import torch
import torch.nn as nn

class GPT2RewardModel(RewardModel):

    def __init__(self, hf_id='gpt2', device='cpu', cache_dir=None):
        super().__init__()
        self.reward_model = GPT2FeedbackScorer(hf_id=hf_id, train=False, device=device, cache_dir=cache_dir)
