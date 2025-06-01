import os
import gc
import sys

import tqdm
import transformers as trf
import torch as tc
import numpy as np



trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["OMP_NUM_THREADS"] = '10'
CACHE_DIR = "/data/Huggingface"


class Generator:
    def __init__(self, model_name, device="cuda"):
        self._name = model_name
        self._device = device 
        self.build_model()

    @tc.no_grad()
    def build_model(self):
        print("Initializing LLM: %s" % self._name) 
        maps = "cpu" if self._device == "cpu" else "auto"
        self._tokenizer = trf.AutoTokenizer.from_pretrained(self._name, use_fast=False, padding_side="right", cache_dir=CACHE_DIR)
        self._model = trf.AutoModelForCausalLM.from_pretrained(self._name, cache_dir=CACHE_DIR, device_map=maps, attn_implementation="eager", torch_dtype=tc.bfloat16)
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

    @tc.no_grad()
    def generate(self, text, **kwrds):
        ids = tc.tensor([self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))]).to(self._device)
        tokens = self._model.generate(ids, pad_token_id=self._tokenizer.eos_token_id, **kwrds)[:, ids.shape[1]:]
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    def get_activates(self, ids, return_logits=False):
        if isinstance(ids, str):
            assert ids.startswith("<s>")
            ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(ids))
        ids = tc.tensor([ids[: 2048]]).to(self._device)
        outs = self._model(ids, output_hidden_states=True, return_dict=True)
        if not return_logits:
            return outs.hidden_states
        return outs.hidden_states, outs.logits


if __name__ == "__main__":

    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    print(generator.generate("<s>[INST] Please explain for me: 9.11 and 9.9, who is larger? [/INST]", max_new_tokens=128, do_sample=False))
    print(generator.generate("[INST] Please explain for me: 9.11 and 9.9, who is larger? [/INST]", max_new_tokens=128, do_sample=False))
