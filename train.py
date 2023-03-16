import colored_traceback.always
import pprint
import torch
import torch.nn as nn
from models.GPT2ActorCriticNetwork import GPT2ActorCriticNetwork
from models.GPT2ActorCriticPolicy import GPT2ActorCriticPolicy
from envs.AdaptEnv import AdaptEnv
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models.GPT2RewardModel import GPT2RewardModel
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
from datetime import datetime

# device = 'cpu'
device = 'cuda:0'

if device == 'cpu':
    print('WARNING: training on cpu.')

cache_dir = '/om2/user/rogerjin/.cache/transformers'

import os
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['WANDB_CACHE_DIR'] = cache_dir

hf_models = {
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'DialoGPT-small': 'microsoft/DialoGPT-small',
    'DialoGPT-medium': 'microsoft/DialoGPT-medium',
    'DialoGPT-large': 'microsoft/DialoGPT-large'
}

learning_algos = {
    'ppo': PPO,
    'a2c': A2C
}

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config_path = args.config

config = json.loads(open(config_path).read())
config.setdefault('temperature', 0.6)

print(config)

# config = {
#     'out_type': 'code',
#     'prompt': EOS_TOKEN,
#     'feedback': ";",
#     'negative_feedback': False,
#     'emotion': 'semicolon',
#     'model_name': 'gpt2-medium',
#     'algorithm': 'ppo',
#     'max_seq_len': 35,
#     'max_steps': 50000,
#     'lr': 1e-6, # from https://github.com/openai/lm-human-preferences
#     'n_envs': 40,
#     'save_freq': 10000,
#     'gen_freq': 1000,
# }

TOY = False
# TOY = True
config['toy'] = TOY
if TOY:
    print('TOY')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    config['max_steps'] = 4
    config['max_seq_len'] = 10
    config['algorithm'] = 'a2c'
    config['n_envs'] = 2
    config['save_freq'] = 2
    config['gen_freq'] = 1
    config['model_name'] = 'gpt2'

config['policy_kwargs'] = {
    'ortho_init': False, # important - use pretrained weights
    'policy_hfid': hf_models[config['model_name']],
    'value_hfid': hf_models[config['model_name']],
    'device': device,
    'cache_dir': cache_dir,
}

today = datetime.today().strftime('%Y-%m-%d')
curr_time = datetime.now().strftime('%H-%M-%S')

save_base_dir='checkpoints'
save_rel_dir=f"{config['out_type']}/{config['emotion']}/{config['model_name']}/{today}/{curr_time}"
save_dir = f'{save_base_dir}/{save_rel_dir}' if not TOY else 'test/checkpoints'

pprint.pprint(config, indent=2)

run = wandb.init(project="adaPT", entity="6-884-adapt", config=config, sync_tensorboard=True, monitor_gym=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>', device=device, cache_dir=cache_dir)
prompt_len = tokenizer(config['prompt'], return_tensors="pt")['input_ids'].shape[-1]

reward_model = GPT2RewardModel(device=device) # todo: try out dialogpt?

env_kwargs = {
    'prompt': config['prompt'],
    'feedback': config['feedback'],
    'max_seq_len': config['max_seq_len'],
    'negative_feedback': config['negative_feedback'],
    'reward_model': reward_model,
    'tokenizer': tokenizer,
    'eos_token_id': tokenizer.eos_token_id,
    'device': 'cpu', # i think breaks on cuda rn
}

if config['n_envs'] == 1:
    env = AdaptEnv(**env_kwargs)
else:
    env = make_vec_env(AdaptEnv, n_envs=config['n_envs'], env_kwargs=env_kwargs)
# eval_env = AdaptEnv(config['prompt'], config['feedback'], 35, reward_model , tokenizer, tokenizer.eos_token_id, device='cpu')

learning_algo = learning_algos[config['algorithm']] # default PPO. If TOY, then A2C.

n_steps = config['max_seq_len'] - prompt_len
if not TOY:
    model = learning_algo(GPT2ActorCriticPolicy, env, n_steps=n_steps, learning_rate=config['lr'], batch_size=config['n_envs'], policy_kwargs=config['policy_kwargs'], device=device, verbose=2, tensorboard_log=f"tensorboard/{run.id}")
else:
    model = learning_algo(GPT2ActorCriticPolicy, env, n_steps=n_steps, learning_rate=config['lr'], policy_kwargs=config['policy_kwargs'], device=device, verbose=2, tensorboard_log=f"tensorboard/{run.id}")
model.policy._modules['action_net'] = GPT2LMHeadModel.from_pretrained(hf_models[config['model_name']]).to(device).lm_head
print('training')

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps

from AdaptCallbacks import GenerateCallback

save_name = f"{config['model_name']}_{config['algorithm']}_{config['max_steps']}_lr={config['lr']}"

checkpoint_callback = CheckpointCallback(save_freq=config['save_freq'], save_path=save_dir, name_prefix=save_name)

num_gen_seqs = 20 if not TOY else 2
sentences = [config['prompt']] * num_gen_seqs

gen_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>', device=device, cache_dir=cache_dir)
gen_root_dir = 'output' if not TOY else "test/output"
generate_callback = GenerateCallback(sentences, save_rel_dir, save_name, gen_tokenizer, wandb, save_root_dir=gen_root_dir, device=device)
periodic_gen_callback = EveryNTimesteps(config['gen_freq'], generate_callback)

model.learn(config['max_steps'], callback=[WandbCallback(verbose=2), checkpoint_callback, periodic_gen_callback])
wandb.log({'generations': generate_callback.table})

print('done training, saving...')
model.policy._modules['action_net'].bias = torch.nn.Parameter(torch.Tensor([0]*50257).to(device))
save_path = f'{save_dir}/{save_name}.zip'
model.save(save_path)
print('saved to:', save_path)

model = model.policy._modules['mlp_extractor'].policy_latent.lm # todo: figure out if the weights are different

out_dir = f'output/{save_rel_dir}' if not TOY else f'test/output/{save_rel_dir}'
os.makedirs(out_dir, exist_ok=True)
out_path = f'{out_dir}/{save_name}.txt'

# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

num_sequences = 1000
batch_size = 1
if TOY:
    num_sequences = 4
    batch_size = 1
sentences = [config['prompt']] * num_sequences

output_sequences = []

for i in range(0, num_sequences, batch_size):
    batch = sentences[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True)

    output_sequences.extend(model.generate(
        input_ids=inputs['input_ids'].to(device),
        attention_mask=inputs['attention_mask'].to(device),
        do_sample=True, # disable sampling to test if batching affects output
        min_length=25,
        max_length = 60,
        temperature=config['temperature'],
        top_k=50,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
    )) # https://github.com/microsoft/DialoGPT/issues/45

decodings = []
decodings_formatted = []

for i in range(len(sentences)):
    decoding = tokenizer.decode(output_sequences[i], skip_special_tokens=True)
    if TOY:
        print(decoding)
    decodings.append(decoding)
    decodings_formatted.append(f'<|DECODING|>{decoding}')
#     decodings.append(f'{decoding}\n')
    # you can use skip_special_tokens=True in decode() to remove padding token
    # but note that it will also remove other special_tokens

with open(out_path, 'w+') as out:
    out.writelines(decodings_formatted)

columns = ['generation']
table = wandb.Table(data=[[d] for d in decodings], columns=columns)
wandb.log({'final_generations': table})

run.finish()
