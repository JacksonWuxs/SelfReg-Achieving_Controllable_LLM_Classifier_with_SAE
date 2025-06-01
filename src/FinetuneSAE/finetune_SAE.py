import sys
import os
import psutil
import time
import pickle as pkl

import tqdm
import torch as tc
import numpy as np
import transformers as trf

from corpus import CorpusSearchIndex
from generator import Generator
from autoencoder3 import load_pretrained, MaskTopK, normalized_l2
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


def identify_dead_neurons(sae, shift, corpus, reload=False):
    if reload and os.path.exists("./TopK7_l%d_h65k_%s_dead.pkl" % (layer, dataset)):
        return pkl.load(open("./TopK7_l%d_h65k_%s_dead.pkl" % (layer, dataset), "rb"))
        
    sae.alpha = 0
    losses = []
    bar = tqdm.tqdm(total=len(corpus), desc="CountDead")
    freq = tc.zeros(sae.dims[1])
    with tc.no_grad():
        with tc.autocast(device_type="cuda", dtype=tc.bfloat16, enabled=True):
            for actv in corpus.get_data():
                a = sae._encode(actv[:1024].cuda())
                freq += tc.where(a > 0, 1., 0.).sum(axis=0).cpu()
                r = sae._decode(tc.relu(a[shift:] * MaskTopK(a[shift:], sae.topk)))
                losses.append(normalized_l2(r, actv[shift:]).item())
                bar.update(1)
    deads = [i for i, v in enumerate(freq.tolist()) if v == 0]
    print("Totally detected dead neurons: %d." % len(deads))
    print("Average Reconstruction Loss: %.4f." % np.mean(losses))
    if reload:
        pkl.dump(deads, open("./TopK7_l%d_h65k_%s_dead.pkl" % (layer, dataset), "wb"))
    return deads


def testify(sae, model):
    sae.eval()
    switch_mode(sae, "generate")
    with tc.autocast(device_type="cuda", dtype=tc.bfloat16, enabled=True):
        text = model.generate("<s>[INST] Who is the president of US? [/INST]", max_new_tokens=128, do_sample=False)
    sae.train()
    switch_mode(sae, "train")
    return text


def finetune_SAE(corpus, sae, generator, shift, batch_size=512, learn_rate=5e-5, epochs=5, betas=(0.9, 0.999)):
    optimizer = tc.optim.Adam(sae.parameters(), lr=learn_rate, betas=betas, eps=6.25e-10, weight_decay=0.0)
    scaler = tc.cuda.amp.GradScaler(enabled=True)
    batch_size = min(batch_size, len(corpus)) 
    sae.alpha = 0.0
    skips = (1, 0)
    freq = max(1000, batch_size)
    for epoch in range(1, 1 + epochs):
        print("\n\n")
        print("Epoch=%d" % epoch, "-->", testify(sae, generator))
        bar = tqdm.tqdm(total=len(corpus), desc="Epoch=%d" % epoch)
        ttls, l1s, l2s, l0s = [], [], [], []
        corpus.shuffle(epoch)
        if epoch >= skips[0]:
            print("Epoch=%d " % epoch, end='')
            sae.set_dead_mask(identify_dead_neurons(sae, shift, corpus, epoch==1))
            for actv in corpus.get_data():
                bar.update(1)
                if bar.n <= skips[1]:
                    continue
                with tc.autocast(device_type="cuda", dtype=tc.bfloat16, enabled=True):
                    ttl, l0, l1, l2 = sae.compute_finetune(actv[shift:1024].cuda())
                scaler.scale(ttl).backward()
                
                ttls.append(sae.ttl.item())
                l0s.append(sae.l0.item())
                l1s.append(sae.l1.item())
                l2s.append(sae.l2.item())

                if  bar.n % batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if bar.n % freq == 0:
                    print("\nEpoch=%d | Step=%d | Total=%.4f | L0=%.4f | L1=%.4f | L2=%.4f" % (
                    epoch, bar.n, np.mean(ttls[-freq:]), np.mean(l0s[-freq:]), np.mean(l1s[-freq:]), np.mean(l2s[-freq:])))
                    ttls.clear(); l0s.clear(); l1s.clear(), l2s.clear()

                if bar.n % (25 * batch_size) == 0:
                    sae.dump_disk("outputs/TopK7_l%d_h65k_FT_epoch%d_%s.pth" % (layer, epoch, dataset))
                    print("Epoch=%d | Step=%d" % (epoch, bar.n), "-->", testify(sae, generator))
        bar.close()
    return sae




def is_process_running(pid):
    """Check if a process with a given PID is still running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def wait_for_programs_to_finish(pid_list):
    """Wait until all programs in the PID list have completed."""
    while any(is_process_running(pid) for pid in pid_list):
        print("Waiting for programs to finish...")
        time.sleep(60)  # Wait for 5 seconds before checking again



if __name__ == "__main__":
    name, layer, sae = load_pretrained(sys.argv[2])
    dataset, shift, epochs = sys.argv[3], sys.argv[4], int(sys.argv[5])
    print("Dataset=%s | Shifting: %s" % (dataset, shift)) 
    corpus = GroupActvDataset("./datasets/%s_train/" % dataset, layerID=layer) 
    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    print("Golden:", generator.generate("<s>[INST] Who is the president of US? [/INST]", max_new_tokens=128, do_sample=False))
    mount_function(generator._model, "mistral", int(layer), sae)
    finetune_SAE(corpus, sae, generator, int(shift), epochs=epochs)
    
