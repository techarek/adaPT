from stable_baselines3.common.callbacks import BaseCallback
import os

class GenerateCallback(BaseCallback):

    def __init__(self, input_sentences, save_rel_dir, save_name, tokenizer, wandb, save_root_dir='output', batch_size=1, device='cpu', verbose=True):
        super().__init__()
        self.save_rel_dir = save_rel_dir
        self.save_root_dir = save_root_dir
        self.save_name = save_name
        self.out_dir = f'{self.save_root_dir}/{self.save_rel_dir}'
        self.input_sentences = input_sentences
        self.num_sequences = len(input_sentences)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = tokenizer.eos_token # to avoid an error
        self.wandb = wandb
        self.table = self.wandb.Table(columns=['step', 'generation'])

        self.lm = None
        self.device = device
        self.verbose = verbose

    def _init_callback(self):
        return

    def _on_step(self):

        print('='*100)
        print(f'STEP {self.num_timesteps}'.center(100))
        print('='*100)

        if self.lm is None:
            self.lm = self.model.policy._modules['mlp_extractor'].policy_latent.lm # todo: figure out if the weights are different

        save_rel_dir = self.save_rel_dir
        sentences = self.input_sentences
        num_sequences = self.num_sequences
        batch_size = self.batch_size
        model = self.lm
        device = self.device
        tokenizer = self.tokenizer

        out_dir = self.out_dir

        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/{self.save_name}_{self.num_timesteps}.txt'

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
                temperature=0.6,
                top_k=50,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
            )) # https://github.com/microsoft/DialoGPT/issues/45

        decodings = []
        decodings_formatted = []

        for i in range(len(sentences)):
            decoding = tokenizer.decode(output_sequences[i], skip_special_tokens=True)
            if self.verbose:
                print(f'GEN {i}:', decoding)
            decodings.append(decoding)
            decodings_formatted.append(f'<|DECODING|>{decoding}')
            self.table.add_data(self.num_timesteps, decoding)
        #     decodings.append(f'{decoding}\n')
            # you can use skip_special_tokens=True in decode() to remove padding token
            # but note that it will also remove other special_tokens

        with open(out_path, 'w+') as out:
            out.writelines(decodings_formatted)

        return True
