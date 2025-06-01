import time
import re
import os
import string
import json
import concurrent
import multiprocessing

import pandas as pd
import transformers as trf
import tqdm


CACHE_DIR = "../../"


class Reader:
    def __init__(self, fpath):
        self.df = pd.read_csv(fpath, engine="python", 
    on_bad_lines="skip", encoding="utf8", sep="\t")
        print("Loading success!")
        self.df.sort_values("NeuronID", inplace=True)
        self.tokenizer = trf.AutoTokenizer.from_pretrained(
                           "mistralai/Mistral-7B-Instruct-v0.2", 
                           use_fast=False, padding_side="right", 
                           cache_dir=CACHE_DIR)

    def select(self, idx, topK=5, key="Span"):
        i = self.df.NeuronID.searchsorted(idx, side="left")
        j = self.df.NeuronID.searchsorted(idx, side="right")
        if not i <= j - 1:
            return []
        df = self.df.iloc[i:j]
        df = df.sort_values(by="Score", ascending=False)
        return df[key].tolist()[:topK]

    def truncate(self, span, topN=10):
        if not isinstance(span, str):
            span = ''
        ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(span))[-topN:]
        return self.tokenizer.batch_decode([ids])[0]

    def get_neuron_spans(self, idx, topK, topN=10):
        spans = [self.truncate(_, topN) for _ in self.select(idx, topK)]
        return "\n".join("Span %d: %s" % pair
                         for pair in enumerate(spans, 1))
        

if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    print("Grouping By Files from: %s" % folder)
    
    duplicated = set()
    with open(folder + "/full_deduplicated.tsv", "w", encoding="utf8") as f:
        with open(folder + "/full.tsv", encoding="utf8") as g:
            f.write(g.readline())
            for row in g:
                temp = row.split("\t")
                temp = (temp[0], temp[-1])
                if temp in duplicated:
                    continue
                f.write(row)
                duplicated.add(temp)

    reader = Reader(folder + "/full.tsv")
    print("Loading %d deduplicated records." % len(reader.df))
    file = os.path.split(folder)[-1]

    bar = tqdm.tqdm(total=2**16)
    activated = 0
    print("./%s.tsv" % file.replace("textspans", "TopAct"))
    with open("./%s.tsv" % file.replace("textspans", "TopAct"), "w", encoding="utf8") as f:
        f.write("FeatureID\tWords\n")
    #if True:
        for idx in range(2 ** 16):
            span = reader.get_neuron_spans(idx, topK=10)
            if "Span" in span:
                activated += 1
            f.write("%d\t%s\n" % (idx, span.replace("\t", "\\t").replace("\n", "\\n").replace("\r", ""))) 
            bar.update(1)
    print("Totally %d neurons are activated." % activated)
        
        
        

