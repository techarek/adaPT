import torch
import torch.nn as nn
from models.ValueNetwork import ValueNetwork
from models.GPT2FeedbackScorer import GPT2FeedbackScorer

class GPT2ValueNetwork(ValueNetwork):

    def __init__(self, hf_id, device='cpu', cache_dir=None):
        super().__init__()
        self.feedback = GPT2FeedbackScorer(hf_id, device=device, cache_dir=cache_dir) # todo: this is super lazy, maybe do something like GPT2Model --> linear
        #self.model = lambda x: 0.3

    def forward(self,tokenized_text):
        return self.feedback(tokenized_text) # I think this gives a log prob
