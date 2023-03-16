import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class AdaptActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, build_network_fn, *args, **kwargs):
        self.build_network_fn = build_network_fn
        kwargs.pop('use_sde')
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = self.build_network_fn()



