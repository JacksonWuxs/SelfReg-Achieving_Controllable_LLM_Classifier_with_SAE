import os
import sys
import tqdm
import pickle as pkl
import numpy as np
import torch as tc
import transformers as trf
import datasets


trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
CACHE_DIR = "/data/Huggingface/"

from corpus import CorpusSearchIndex
from generator import Generator
from llm_surgery import mount_function, switch_mode


def collect_actvs(prompt, model, collectors):
    try:
        model.get_activates(prompt)
    except RuntimeError:
        pass
    for collector in collectors:
        yield collector.cache.cpu().to(tc.float16).detach().squeeze()


class Collector:
    def __init__(self, layer,):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


if __name__ == "__main__":
    template = "<s>[INST] %s [/INST]"
    corpus = CorpusSearchIndex("datasets/prompt_dataset_train.tsv", cache_freq=10000, sampling=None)
    model = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    group_size = 256 
    group_idx = int(sys.argv[1])
    jump = int(sys.argv[2])

    layers = [8, 16, 24, 32]
    group_actvs = []
    collectors = []
    for layer in layers:
        collectors.append(Collector(layer))
        group_actvs.append([])
        mount_function(model._model, "mistral", layer, collectors[-1])
    collectors[-1].early_stop = True

    with tc.no_grad():
        bar = tqdm.tqdm(total=len(corpus))
        for i, text in enumerate(corpus):
            bar.update(1)
            try:
                if i < group_idx * group_size:
                    continue
                fpath = "datasets/prompt_actvs_l%d/group_%d.pt" % (layers[0], group_idx)
                if os.path.exists(fpath):
                    group_idx += jump
                    continue
                prompt = template % text
                for g_actv, item in zip(group_actvs, collect_actvs(prompt, model, collectors)):
                    g_actv.append(item)
                if len(group_actvs[0]) == group_size:
                    for l, g_actv in zip(layers, group_actvs):
                        fpath = "datasets/prompt_actvs_l%d/group_%d.pt" % (l, group_idx)
                        tc.save(g_actv, fpath)
                        g_actv.clear()
                    group_idx += jump
            except Exception as e:
                print("Encounter Error at line %d: %s" % (i, e))
        if len(group_actvs) > 0:
            for l, g_actv in zip(layers, group_actvs):
                fpath = "datasets/prompt_actvs_l%d/group_%d.pt" % (l, group_idx)
                tc.save(g_actv, fpath)
        print("Stopping with %s tokens." % bar.n)
    




