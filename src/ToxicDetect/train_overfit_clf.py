import os
import sys

import torch as tc
import numpy as np
import tqdm


from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE
from train_simple_clf import LinearRewarder


class LinearRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "OverfitRewarder")
        self.max_tolerate = self.epochs = 70
        self.reload_best = False


if __name__ == "__main__":
    config = LinearRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=True)
    sae = TopKSAE.from_disk("../TopK6_l16_h65k_epoch5.pth")
    Actv = sae._encode(model.linear1.weight).squeeze().cpu().tolist()
    chosen1, chosen2, chosen3 = 0, 0, 0
    fpath = "annotations/TopAct_TopK6_l16_h65k_epoch5_smallRWBench_frequent.tsv"
    with open(fpath, encoding="utf8") as f,\
         open(fpath.rsplit(".", 1)[0] + "seed%d_overfit.tsv" % config.seed, "w", encoding="utf8") as g:
        g.write("Actv\t" + f.readline())
        rows = [0] * 65536
        for row in f:
            rows[int(row.split("\t")[1])] = row
        for i, (row, actv) in enumerate(zip(rows, Actv)):
            if row == 0 or "Span" not in row:
                g.write("%.4f\t0\t%d\n" % (actv, i))
            else:
                g.write("%.4f" % actv + '\t' + row)
                chosen3 += 1
            if actv > 0.1:
                chosen1 += 1
                if row != 0 and float(row.split("\t")[0]) > 0.5:
                    chosen2 += 1
    print("Chosen1=%d | Chosen2=%d | Chosen3=%d" % (chosen1, chosen2, chosen3))
