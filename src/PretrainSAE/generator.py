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
CACHE_DIR = "/data/Huggingface"



class Generator:
    def __init__(self, model_name, device="cuda", dtype="float32"):
        self._name = model_name
        self._device = device 
        self._dtype = dtype
        self.build_model()

    @tc.no_grad()
    def build_model(self):
        print("Initializing LLM: %s" % self._name) 
        maps = "cpu" if self._device == "cpu" else "auto"
        self._tokenizer = trf.AutoTokenizer.from_pretrained(self._name, use_fast=True, padding_side="right", cache_dir=CACHE_DIR)
        self._model = trf.AutoModelForCausalLM.from_pretrained(self._name, cache_dir=CACHE_DIR, torch_dtype=self._dtype, device_map=maps)
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id
        self._inp_emb = self._model.get_input_embeddings()
        self._out_emb = self._model.get_output_embeddings()
        self._out_norm = self._model.base_model.norm

    @tc.no_grad()
    def generate(self, text, **kwrds):
        ids = tc.tensor([self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))]).to(self._device)
        tokens = self._model.generate(ids, pad_token_id=self._tokenizer.eos_token_id, **kwrds)[:, ids.shape[1]:]
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    def get_activates(self, ids, return_logits=False):
        if isinstance(ids, str):
            ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
        outs = self._model(tc.tensor([ids[:512]]).to(self._device), output_hidden_states=True, return_dict=True)
        if not return_logits:
            return outs.hidden_states
        return outs.hidden_states, outs.logits


instruct = "You are a neuron interpreter for neural networks. Each neuron looks for one particular concept/topic/theme/behavior/pattern. " +\
               "Look at some words the neuron activates for and summarize in a single concept/topic/theme/behavior/pattern what the neuron is looking for. " +\
               "The words close to the beginning are more correlated to the hidden concept/topic/theme/behavior/pattern. " +\
               "Don't list examples of words and keep your summary as concise as possible. " +\
               "If you cannot summarize more than half of the given words within one clear concept/topic/theme/behavior/pattern, you should say ``Cannot Tell.``."
examples = [
                ("January, terday, cember, April, July, September, December, Thursday, quished, November, Tuesday.",
                 "Dates."),
                ("B., M., e., R., C., OK., A., H., D., S., J., al., p., T., N., W., G., a.C., or, St., K., a.m., L..",
                 "Abbrevations and acronyms."),
                ("actual, literal, real, Real, optical, Physical, REAL, virtual, visual.",
                 "Perception of reality."),
                ("Go, Python, C++, Java, c#, python3, cuda, java, javascript, basic.",
                 "Programing languages."),
                ("1950, 1980, 1985, 1958, 1850, 1980, 1960, 1940, 1984, 1948.",
                 "Years."),
                ]
        
if __name__ == "__main__":

    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    print(generator.generate("<s>[INST] Who is the current president of United State? [/INST]", max_new_tokens=128, do_sample=False))
