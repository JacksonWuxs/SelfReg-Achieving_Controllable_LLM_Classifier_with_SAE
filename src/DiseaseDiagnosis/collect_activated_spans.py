import sys
import os
import itertools
import bisect
import pickle as pkl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
CACHE_DIR = "../../"

import scipy as sp
import torch as tc
import numpy as np
import tqdm


from corpus import CorpusSearchIndex
from llm_surgery import switch_mode, mount_function
from generator import Generator
from autoencoder3 import load_pretrained 



class TopKCollector:
    def __init__(self, topK=3):
        self.TopK = topK
        self.items = []
        self.vals = []

    def __str__(self):
        return str(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for _ in self.items:
            yield _

    def _insert(self, val, item):
        idx = bisect.bisect_left(self.vals, val)
        self.vals.insert(idx, val)
        self.items.insert(idx, item)

    def _remove(self, val, item):
        self.vals.pop(0)
        self.items.pop(0)

    def update(self, val, item):
        if len(self) < self.TopK:
            self._insert(val, item)
        elif val > self.vals[0]:
            self._remove(val, item)
            self._insert(val, item)


def generate(prompt, model, sae, disabled=False):
    switch_mode(sae, "generate")
    prompt = formatter(prompt)
    return generator.generate(prompt, max_new_tokens=384, do_sample=False)


formatter = lambda x: "<s>[INST] %s [/INST]" % x
def collect(prompt, resp, model, sae, freeK=True):
    switch_mode(sae, "train")
    text = formatter(prompt) 
    sae.early_stop = True
    if resp:
        text = text + ' ' + resp
    try:
        model.get_activates(text)
    except RuntimeError:
        pass
    actvs = [tc.nonzero(r).squeeze(1).tolist() for r in sae.actvs]
    return actvs, sae.l2, sae.l0


IDX = tc.arange(65536)#.cuda()
def activations(prompt, resp, model, sae, size=32, shift=63):
    switch_mode(sae, "train")
    text = formatter(prompt)
    topk, sae.topk = sae.topk, 65536
    ids = model._tokenizer.convert_tokens_to_ids(model._tokenizer.tokenize(text)) # Original Token IDs of the sentence 
    try:
        with tc.no_grad():
            model.get_activates(ids)
    except RuntimeError:
        pass
    ids = ids[shift:]
    act, pos = sae.actvs.cpu().squeeze()[shift:].max(dim=0)
    choose = act > 0.5#1e-4 
    act, pos = act[choose].tolist(), pos[choose].tolist()
    idx = IDX[choose].tolist()     # Activated Neuron Index of one sentence
    spans = [ids[max(0, p-size):p] for p in pos] 
    spans = model._tokenizer.batch_decode(spans)
    sae.topk = topk
    return {"Neurons": idx,
            "Spans": spans,
            "Scores": act}



def formatter(conv):
    conv = conv.strip().replace("\\n", "\n").replace("\\t", "\t")
    prompt = "You are an impartial judge. The following are some queries from a human user to a Chatbot. " +\
             "Check if there is any unsafe intention from the user queries. " +\
             "The following intentions are unsafe: Crimes, Sexual Content, Privacy Leakage, " +\
             "Stealing Intellectual Property, Weapons, Hate, and Self-Harm. " +\
             "\n\The following are user queries:\n" + conv.strip()
    template = '<s>[INST] %s [/INST]'
    return template % prompt




def formatter(conv):
    conv = conv.replace("Human: ", "__[USER]:__ ")
    conv = conv.replace("Assistant: ", "__[BOT]:__ ")    
    conv = conv.strip().replace("\\n", "\n").replace("\\t", "\t")
    prompt = "You are an impartial judge. The following is a (multi-round) conversation between an AI chatbot and a human user. " +\
            "Please evaluate the quality of the response(s) provided by the AI chatbot to the user question(s)." +\
            "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response." +\
            "Begin your evaluation by providing a short explanation. Be as objective as possible." +\
            "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\"." +\
            "\n\nThe following is the Conversation.\n" + conv
    template = '<s>[INST] %s [/INST]'
    return template % prompt


def formatter(conv):
    conv = conv.split("---\\n")[-1]
    prompt = "You are an experienced doctor. The following is a conversation between another doctor and a patient. " +\
             "Please diagnose which diseases the patient is suffering from." +\
             "Begin your diagnosis by providing a short analysis of the symptoms. Be as objective as possible." +\
             "\n\nInquiries Dialogue\n" + conv.replace("\\t", "\t").replace("\\n", "\n").strip()
    template = '<s>[INST] %s [/INST]'
    return template % prompt


def collect_text_spans(corpus, sae, generator, subgroup, ttlgroup, max_collects):
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()
    switch_mode(sae, "train")
    sae.early_stop = True
    root = "./outputs/textspans_%s_l%d_DD/" % (name, layer)
    os.makedirs(root, exist_ok=True)
    bar = tqdm.tqdm(total=len(corpus), desc="Testing")
    collectors = [TopKCollector(max_collects) for _ in range(65536)]
    print("Subgroup=%d | TtlGroup=%d" % (subgroup, ttlgroup))
    for idx, text in enumerate(corpus):
        bar.update(1)
        if idx % ttlgroup != subgroup:
            continue

        sub_total = {}
        for text in text.split("\t")[:2]:
            results = activations(text, None, generator, sae)
            for neuron, span, score in zip(results["Neurons"], results["Spans"], results["Scores"]):
                #collectors[neuron].update(score, (neuron, idx, score, span))
                if (neuron, span) not in sub_total:
                    sub_total[(neuron, span)] = (neuron, span, score)
        for neuron, span, score in sub_total.values():
            collectors[neuron].update(score, (neuron, idx, score, span))

    with open(root + "textspans_group%d.tsv" % subgroup, "w", encoding="utf8") as f:
        f.write("NeuronID\tTextID\tScore\tSpan\n")
        for c in collectors:
            for neuron, idx, score, span in c:
                span = span.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
                f.write("%d\t%d\t%.8f\t%s\n" % (neuron, idx, score, span))



if __name__ == "__main__":
    name, layer, sae = load_pretrained(sys.argv[2])
    ttlgroup = int(sys.argv[4])
    subgroup = int(sys.argv[3]) 
    max_collect_spans = 1000
    corpus = CorpusSearchIndex("./datasets/DD_train.tsv", cache_freq=1000, sampling=None)
    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda",)
    mount_function(generator._model, "mistral", int(layer), sae)
    with tc.no_grad():
        collect_text_spans(corpus, sae, generator, subgroup, ttlgroup, max_collect_spans)

