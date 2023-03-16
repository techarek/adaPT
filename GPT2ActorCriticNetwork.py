import torch
import torch.nn as nn

from models.GPT2ValueNetwork import GPT2ValueNetwork
from models.GPT2PolicyNetwork import GPT2PolicyNetwork
from transformers import GPT2Tokenizer

class GPT2ActorCriticNetwork(nn.Module):

    def __init__(self, policy_hfid='gpt2', value_hfid='gpt2', device='cpu', cache_dir=None):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')
        self.policy_latent = GPT2PolicyNetwork(policy_hfid, device=device, cache_dir=cache_dir)
        self.value_latent = GPT2ValueNetwork(value_hfid, device=device, cache_dir=cache_dir)
        self.device = device
        self.latent_dim_pi = self.policy_latent.lm.config.n_embd # 768 for small
        self.latent_dim_vf = self.value_latent.feedback.model.config.n_embd

    def forward_actor(self, state):
        return self.policy_latent(state)

    def forward_critic(self, state):
        return self.value_latent(state)

    def forward(self, state):
        actor = []
        critic = []
        state = state.long()
        for i in range(state.shape[0]):
          x = (state[i] == self.tokenizer.pad_token_id).nonzero(as_tuple=True)
          if x[0].shape[0] != 0:
            state_c = state[i,:x[0][0]]
          else:
              state_c = state[i, :]
          state_c = state_c.unsqueeze(0)
          actor.append(self.forward_actor(state_c))
          critic.append(self.forward_critic(state_c))
        return torch.cat(actor).to(self.device), torch.cat(critic).to(self.device)
