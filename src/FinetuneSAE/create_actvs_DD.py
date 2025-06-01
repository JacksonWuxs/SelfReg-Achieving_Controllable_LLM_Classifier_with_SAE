import os
import sys
import tqdm
import pickle as pkl
import numpy as np
import torch as tc
import transformers as trf
import datasets
from huggingface_hub import login


trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
CACHE_DIR = "./"

login("PUT_YOUR_HF_TOKEN_HERE")

from corpus import CorpusSearchIndex
from generator import Generator
from llm_surgery import mount_function, switch_mode




def collect_actvs(prompt, model, collectors):
    try:
        model.get_activates(prompt)
    except RuntimeError:
        pass
    actvs = {}
    for collector in collectors:
        actv = collector.cache.cpu().to(tc.float16)#.numpy()
        print(actv.shape)
        raise
        actvs[collector.layer] = actv.squeeze(0)#[-1:]
    return actvs


class Collector:
    def __init__(self, layer,):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


def formatter(conv):
    conv = conv.split("---\\n")[-1]
    prompt = "You are an experienced doctor. The following is a conversation between another doctor and a patient. " +\
             "Please diagnose which diseases the patient is suffering from." +\
             "Begin your diagnosis by providing a short analysis of the symptoms. Be as objective as possible." +\
             "\n\nInquiries Dialogue\n" + conv.replace("\\t", "\t").replace("\\n", "\n").strip()
    template = '<s>[INST] %s [/INST]'
    return template % prompt


LABELS = {
          "Pneumonia": 0,
          "URTI": 1,
          "Upper Respiratory Tract Infection": 1,
          }
if __name__ == "__main__":
    import tqdm
    import pickle as pkl
    corpus = CorpusSearchIndex("datasets/DD_train.tsv", cache_freq=10000, sampling=None)
    model = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    group_actvs = []
    group_size = 4 
    group_idx = int(sys.argv[1])
    jump = int(sys.argv[2])

    layers = [16]
    collectors = []
    for layer in layers:
        collectors.append(Collector(layer))
        mount_function(model._model, "mistral", layer, collectors[-1])
    collectors[-1].early_stop = True

    with tc.no_grad():
        bar = tqdm.tqdm(total=len(corpus))
        for i, text in enumerate(corpus):
            if "\t" not in text:
                continue
            prompt, label = text.split("\t")[:2]
            if label in LABELS:
                label = LABELS[label]
            else:
                label = 2
            prompt = formatter(prompt)
            bar.update(1)
            try:
                if i < group_idx * group_size:
                    continue
                fpath = "datasets/dxy3_mistral_%s/group_%d.pt" % (subset, group_idx)
                if os.path.exists(fpath):
                    group_idx += jump
                    continue
                actv = collect_actvs(prompt, model, collectors)
                group_actvs.append([actv, int(label)])
                if len(group_actvs) == group_size:
                    tc.save(group_actvs, fpath)
                    group_actvs.clear()
                    group_idx += jump
            except Exception as e:
                print("Encounter Error at line %d: %s" % (i, e))
        if len(group_actvs) > 0:
            tc.save(group_actvs, fpath)
        print("Stopping with %s tokens." % bar.n)
    




