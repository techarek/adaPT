import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models.FeedbackScorer import FeedbackScorer

class GPT2FeedbackScorer(FeedbackScorer):

    MASK_OUT = -100

    def __init__(self, hf_id='gpt2', device='cuda:0', train=False, cache_dir=None): # todo: condition on size
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_id,pad_token='<|pad|>', device=device, cache_dir=cache_dir) # todo: consider adding a self.tokenizer to FeedbackScorer, check device
        self.model = GPT2LMHeadModel.from_pretrained(hf_id, pad_token_id=self.tokenizer.pad_token_id, cache_dir=cache_dir).to(device)
        self.device = device
        if not train:
            self.model.eval()
        # self.prompt = self.tokenizer([' After I said this I was told to: '], return_tensors='pt').to(device)['input_ids']
        # self.prompt = self.tokenizer([' '], return_tensors='pt').to(device)['input_ids']
        self.prompt = torch.Tensor([[]]).to(device).long()

    def _get_concat_and_labels(self, text): # todo: do this with prompting
          text = text.long()
          #text_length = text.shape[-1]
          concat = torch.cat([text,self.prompt], dim=1)
          #labels = concat.clone()
          #labels[:, torch.arange(text_length)] = GPT2FeedbackModel.MASK_OUT # 
          return concat#, labels

    def score(self, text):
        #with torch.no_grad():
            concat = self._get_concat_and_labels(text)
            new = self.model.generate(concat,max_length=concat.shape[1]+1)
            return self.model.transformer(new).last_hidden_state[:,-1,:]

    def reward(self, text, feedback):
        with torch.no_grad():
            return self._reward(text, feedback)

    def _reward(self,text,feedback):

        text = text.to(self.device)
        feedback = feedback.to(self.device)
        text = text.long()
        feedback = feedback.long()

        mask = text != self.tokenizer.pad_token_id
        text_zeroed_padding = (text + 1) * mask # the +1 is to make everything positive
        num_cols_all_padding = (text_zeroed_padding.sum(dim=0) == 0).sum().item() # sum rows across batch - 0 cols are those which are padding across all cols.
        max_len = text.shape[-1] - num_cols_all_padding + self.prompt.shape[-1] + feedback.shape[-1]

        rm_input = []
        rm_labels = []
        for i in range(text.shape[0]): # remove padding
          # x = (text[i] == self.tokenizer.pad_token_id).nonzero(as_tuple=True)
          # if x[1].shape[0] != 0:
          #   text[i] = text[:,:x[1][0]]
          # if x[0].shape[0] != 0:
          #   print('hi', text[i])
          #   print(text[i, :x[0][0]])
          #   text[i] = torch.cat([text[i,:x[0][0]], self.prompt, feedback], dim=1)
          #   print(text[i])
          mask = text[i] != self.tokenizer.pad_token_id
          text_zeroed_padding = (text + 1) * mask # the +1 is to make everything positive
          text_trimmed = text[text_zeroed_padding.nonzero(as_tuple=True)]
          pad_length = max_len - text_trimmed.shape[-1] - self.prompt.shape[-1] - feedback.shape[-1]
          padding = torch.Tensor([self.tokenizer.pad_token_id]*pad_length).to(self.device)
          row = torch.cat([text_trimmed, self.prompt[0], feedback[0], padding])
          labels = row.clone()
          if pad_length > 0:
              labels[-torch.arange(pad_length)-1] = GPT2FeedbackScorer.MASK_OUT

          labels[torch.arange(text_trimmed.shape[-1]+self.prompt.shape[-1])] = GPT2FeedbackScorer.MASK_OUT

          rm_input.append(row)
          rm_labels.append(labels)

        rm_input = torch.cat(rm_input, dim=0).long()
        rm_labels = torch.cat(rm_labels, dim=0).long()

        # concat = torch.cat([text,self.prompt,feedback], dim=1).to(self.device)
        # labels = concat.clone().to(self.device)
        # labels[:, torch.arange(text.shape[1]+self.prompt.shape[1])] = GPT2FeedbackScorer.MASK_OUT # 

        return self.model(rm_input, labels=rm_labels)['loss'].item()

    def forward(self, text):
        score = []
        for i in range(text.shape[0]):
          x = (text[i] == self.tokenizer.pad_token_id).nonzero(as_tuple=True)
          text_c = text[i]
          if x[0].shape[0] != 0:
            text_c = text[i,:x[0][0]]
          text_c = text_c.unsqueeze(0)
          score.append(self.score(text_c))
        return torch.cat(score)


