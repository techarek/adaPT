import torch
import torch.nn as nn
from envs.AdaptEnv import AdaptEnv
from models.GPT2ActorCriticNetwork import GPT2ActorCriticNetwork
from models.AdaptActorCriticPolicy import AdaptActorCriticPolicy

class GPT2ActorCriticPolicy(AdaptActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        policy_hfid = kwargs.pop('policy_hfid', 'gpt2')
        value_hfid = kwargs.pop('value_hfid', 'gpt2')
        device = kwargs.pop('device', 'cpu')
        cache_dir = kwargs.pop('cache_dir', None)
        super().__init__(lambda: GPT2ActorCriticNetwork(policy_hfid, value_hfid, device=device, cache_dir=cache_dir),*args, **kwargs)


