import sys
import os

import tqdm
import torch as tc
import numpy as np
import transformers as trf

from corpus import CorpusSearchIndex
from generator import Generator
from autoencoder import TopKSAE, SparseAutoencoder
from llm_surgery import switch_mode, mount_function
from datautils import GroupActvDataset


trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
CACHE_DIR = "./"


class LinearRater:
    def __init__(self, init, final, total):
        self.init = init
        self.final = final
        self.total = total
        self.diff = abs(final - init)
        self.trend = 1. if final > init else -1.
        self.steps = 0

    def get_value(self):
        progress = self.diff * self.steps / self.total
        value = self.init + self.trend * progress
        if self.trend > 0:
            return min(self.final, value)
        return max(self.final, value)

    def step(self, new_step=1):
        self.steps += new_step
        return self.get_value()



def testify(sae, model):
    sae.eval()
    switch_mode(sae, "generate")
    text = model.generate("<s>[INST] Who is the president of US? [/INST]", max_new_tokens=128, do_sample=False)
    sae.train()
    switch_mode(sae, "train")
    return text


def train_SAE(dataset, sae, generator, batch_size=512, learn_rate=1e-3, epochs=5, betas=(0.9, 0.999)):
    optimizer = tc.optim.Adam(sae.parameters(), lr=learn_rate, betas=betas, eps=6.25e-10, weight_decay=0.0)
    scaler = tc.cuda.amp.GradScaler(enabled=True)
    rater1 = LinearRater(init=200, final=20, total=int(len(dataset) / batch_size * 0.5))
    sae.topk = int(rater1.step()) 
    sae.alpha = 0.0
    skips = (1, 0)
    freq = max(1000, batch_size)
    for epoch in range(1, 1 + epochs):
        print("Epoch=%d" % epoch, "-->", testify(sae, generator))
        bar = tqdm.tqdm(total=len(corpus), desc="Epoch=%d" % epoch)
        ttls, l1s, l2s, l0s = [], [], [], []
        corpus.shuffle(epoch)
        if epoch >= skips[0]:
            for actv in corpus.get_data():
                bar.update(1)
                if bar.n <= skips[1]:
                    continue
                with tc.autocast(device_type="cuda", dtype=tc.bfloat16, enabled=True):
                    ttl, l0, l1, l2 = sae.compute_loss(actv[:256].cuda())
                scaler.scale(ttl).backward()
                
                ttls.append(sae.ttl.item())
                l0s.append(sae.l0.item())
                l1s.append(sae.l1.item())
                l2s.append(sae.l2.item())

                if  bar.n % batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    sae.topk = int(rater1.step()) 

                if bar.n % freq == 0:
                    print("\nEpoch=%d | Step=%d | Total=%.4f | L0=%.4f | L1=%.4f | L2=%.4f" % (
                    epoch, bar.n, np.mean(ttls[-freq:]), np.mean(l0s[-freq:]), np.mean(l1s[-freq:]), np.mean(l2s[-freq:])))
                    ttls.clear(); l0s.clear(); l1s.clear(), l2s.clear()

                if bar.n % (25 * batch_size) == 0:
                    sae.dump_disk("outputs/TopK7_l%d_h65k_epoch%d.pth" % (layer, epoch))
                    print("Epoch=%d | Step=%d" % (epoch, bar.n), "-->", testify(sae, generator))
        bar.close()
    return sae



if __name__ == "__main__":
    layer = int(sys.argv[2])
    corpus = GroupActvDataset("./datasets/prompt_actvs_l%d/" % layer, layerID=None) 
    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    print(generator.generate("<s>[INST] Who is the president of US? [/INST]", max_new_tokens=128, do_sample=False))
    sae = TopKSAE(4096, 2**16, topK=20, device="cuda")
    sae.disabled = False
    sae.logging = False
    mount_function(generator._model, "mistral", int(layer), sae)
    train_SAE(corpus, sae, generator)
    
