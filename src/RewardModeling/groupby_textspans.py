import time
import re
import os
import string
import json
import concurrent
import multiprocessing

import pandas as pd
import tqdm


class Reader:
    def __init__(self, fpath):
        self.df = pd.read_csv(fpath, engine="python", 
    on_bad_lines="skip", sep="\t")
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
    
    with open(folder + "/full.tsv", "w", encoding="utf8") as f:
        for i, subfile in enumerate(os.listdir(folder)):
            with open(folder + "/" + subfile, encoding="utf8") as g:
                headline = g.readline()
                if i == 0:
                    f.write(headline)
                for row in g:
                    f.write(row)

    reader = Reader(folder + "/full.tsv")
    print("Loading files success.")
    file = os.path.split(folder)[-1]

    bar = tqdm.tqdm(total=2**16)
    with open("./%s.tsv" % file.replace("textspans", "TopAct"), "w", encoding="utf8") as f:
        for idx in range(2 ** 16):
            span = reader.get_neuron_spans(idx, topK=10)
            f.write("%d\t%s\n" % (idx, span.replace("\t", "\\t").replace("\n", "\\n").replace("\r", ""))) 
            bar.update(1)
        
        
        

