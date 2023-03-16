import torch
import torch.nn as nn
from models.PolicyNetwork import PolicyNetwork
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2PolicyNetwork(PolicyNetwork):

    def __init__(self, hf_id='gpt2', device='cpu', cache_dir=None):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_id, cache_dir=cache_dir, pad_token='<|pad|>')
        self.lm = GPT2LMHeadModel.from_pretrained(hf_id, pad_token_id=self.tokenizer.pad_token_id, cache_dir=cache_dir).to(device)
        #self.model = lambda x: torch.randint(0, self.tokenizer.vocab_size - 1)
    
    def next_latent(self, text):
        #with torch.no_grad():
            # if not text.dtype == torch.long:
            #     text = text.long()
            new = self.lm.generate(text,max_length=text.shape[1]+1)
            return self.lm.transformer(new).last_hidden_state[:,-1,:]
  
    def forward(self,tokenized_text):

        return self.next_latent(tokenized_text) # I think this gives a log prob
