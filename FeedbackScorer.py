import torch
import torch.nn as nn

def default_prompt(generation, feedback):
    return f'I wrote the following "{generation}" and this was the feedback: "{feedback}"'

class FeedbackScorer(nn.Module):

    def __init__(self, prompt_function=default_prompt):
        super().__init__()
        self.lm = None # language model like gpt-2, to be overwritten in subclass
        # todo: is above a bad programming pattern?
        self.prompt_function = prompt_function

    def score(text, feedback):
        # text, feedback must be pretokenized
        raise NotImplementedError



