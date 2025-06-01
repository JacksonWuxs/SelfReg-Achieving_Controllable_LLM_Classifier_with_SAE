import os
import sys

import torch as tc
import numpy as np
import tqdm


from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE


class LinearRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "LinearRewarder")


class LinearRewarder(tc.nn.Module):
    def __init__(self, config):
        tc.nn.Module.__init__(self)
        self.name = config.name
        self.l1, self.l2 = config.l1, config.l2
        self.drop_rate, self.in_dim = config.dropout, config.in_dim
        frozen(config.seed)
        self.dropout = tc.nn.Dropout(config.dropout)
        self.linear1 = tc.nn.Linear(config.in_dim, 1)
        self.lossfn = tc.nn.BCEWithLogitsLoss()

    def forward(self, X):
        H = self.dropout(X.cuda())
        return self.linear1(H).flatten().to(X.device)

    def compute_loss(self, X, Y):
        R = self(X.cuda())
        Y = Y.squeeze().cuda()
        L = self.lossfn(R, Y) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters())
        P = tc.where(R > 0., 1.0, 0.0)
        A = (P == Y).sum() / R.shape[0]
        return L, A, P.detach().cpu().tolist(), Y.detach().cpu().tolist()  

    def dump_disk(self, fpath):
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        tc.save({"weight": self.state_dict(),
            "config": {"in_dim": self.in_dim, 
                       "dropout": self.drop_rate,
                       "l1": self.l1,
                       "l2": self.l2}},
                fpath)
        print("%s is dumped at %s." % (self.name, fpath))

    def load_disk(self, fpath):
        self.load_state_dict(tc.load(fpath)["weight"], strict=True)
        print("%s is loaded from %s." % (self.name, fpath))


if __name__ == "__main__":
    config = LinearRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
    sae = TopKSAE.from_disk("../TopK6_l8_h65k_epoch5.pth")
    Actv = sae._encode(model.linear1.weight).squeeze().cpu().tolist()
    chosen1, chosen2, chosen3 = 0, 0, 0
    fpath = "annotations/TopAct_TopK6_l16_h65k_epoch5_smallRWBench_frequent.tsv"
    with open(fpath, encoding="utf8") as f,\
         open(fpath.rsplit(".", 1)[0] + "seed%d_activated.tsv" % config.seed, "w", encoding="utf8") as g:
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
