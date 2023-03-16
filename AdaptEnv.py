import gym
from gym import spaces
import numpy as np
import torch

# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html


class AdaptEnv(gym.Env):

    def cat_encodings(self,encoding1, encoding2):
        # todo: consider moving this to tokenizer
        x = (encoding1 == self.eos_token_id+1).nonzero(as_tuple=True)
        if x[1].shape[0] == 0: return encoding1
        encoding1[0,x[1][0]] = encoding2
        return encoding1

    def __init__(
        self,
        prompt,
        feedback,
        max_seq_len,
        reward_model,
        tokenizer,
        eos_token_id,
        device="cuda:0",
        negative_feedback=True,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.action_space = spaces.Discrete(tokenizer.vocab_size)

        self.observation_space = spaces.Box(
            low=0,
            high=tokenizer.vocab_size,
            shape=(1, max_seq_len),
            dtype=np.int64,
        )  # don't set max_seq_len too large cuz transformer runtime quadratic in seq len

        self.device = device
        self.reward_model = reward_model  # (generation, feedback) --> reward
        self.tokenizer = tokenizer
        self.prompt = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=max_seq_len
        )['input_ids'].long().to(
            device
        )  # e.g. Tell me a joke

        self.prompt_len = (self.prompt != self.tokenizer.pad_token_id).sum().item()
        self.len = self.prompt_len

        self.feedback = self.tokenizer(feedback, return_tensors="pt")['input_ids'].to(
            device
        )  # possibly not useful to tokenize here
        self.prefix = self.prompt.clone()
        self.eos_token_id = eos_token_id
        #self.reward_model.to(self.device)
        # self.prefix.to(self.device)
        # self.feedback.to(self.device)
        self.negative_feedback = negative_feedback

    def compute_reward(self):  # todo: probably should decode
        return self.reward_model.reward(self.prefix, self.feedback)

    def append_prefix(self,action):
        self.prefix = self.cat_encodings(self.prefix, action)
        self.prefix.to(self.device)

    """
    I like
    I like apples
    I like apples and
    """

    def step(self, action):
        done = (action == self.eos_token_id or self.prefix[0,-1]!=self.eos_token_id+1 or self.len == self.max_seq_len)
        reward = 0
        if done:
            reward = self.compute_reward()  # problem: rewards are sparse
            if not self.negative_feedback:
                reward = -reward
        else:
            self.append_prefix(action)
        self.len += 1
        return self.prefix, reward, done, {}

    def reset(self):
        self.prefix = self.prompt.clone()
        self.len = self.prompt_len
        return self.prefix
