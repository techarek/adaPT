from transformers import GPT2Tokenizer, GPT2LMHeadModel
from stable_baselines3 import PPO, A2C
from models.GPT2ActorCriticPolicy import GPT2ActorCriticPolicy
from envs.AdaptEnv import AdaptEnv
import os

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
parser.add_argument('--ckpt')
parser.add_argument('--out')
args = parser.parse_args()

config_path = args.config
checkpoint_path = args.ckpt
out_name = args.out

config = json.loads(open(config_path).read())
config.setdefault('temperature', 0.6)

print(config)

config['policy_kwargs'] = {
    'ortho_init': False, # important - use pretrained weights
    'policy_hfid': hf_models[config['model_name']],
    'value_hfid': hf_models[config['model_name']],
    'device': device,
    'cache_dir': cache_dir,
}

pprint.pprint(config, indent=2)

prompt = config['prompt']

out_dir = f'output/final/{out_name}'
os.makedirs(out_dir, exist_ok=True)

with open(f'{out_dir}/prompt.txt', 'w+') as prompt_file:
    prompt_file.write(prompt)

cache_dir = '/om2/user/rogerjin/.cache/transformers'
hf_id = hf_models[config['model_name']]
# hf_id = 'gpt2-large'

tokenizer = GPT2Tokenizer.from_pretrained(hf_id, cache_dir=cache_dir, device=device)


# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

def generate(model, out_path, num_sequences=1000):
    sentences = [prompt] * num_sequences
    batch_size = 1

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
            top_k=20,
            no_repeat_ngram_size=2
        ))

    decodings = []

    for i in range(len(sentences)):
        decoding = tokenizer.decode(output_sequences[i], skip_special_tokens=True)
        print(decoding)
        decodings.append(f'<|DECODING|>{decoding}')
    #     decodings.append(f'{decoding}\n')
        # you can use skip_special_tokens=True in decode() to remove padding token
        # but note that it will also remove other special_tokens

    with open(out_path, 'w+') as out:
        out.writelines(decodings)

model = GPT2LMHeadModel.from_pretrained(hf_id, cache_dir=cache_dir).to(device)
out_path = f'{out_dir}/control.txt'
generate(model, out_path)

ppo = PPO.load(checkpoint_path)
model = ppo.policy._modules['mlp_extractor'].policy_latent.lm # todo: figure out if the weights are different
out_path = f'{out_dir}/trained.txt'
generate(model, out_path)
