import os
import sys

import torch as tc
import numpy as np
import tqdm


from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE


class LSRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "LSRewarder")


class LSRewarder(tc.nn.Module):
    def __init__(self, config):
        tc.nn.Module.__init__(self)
        self.name = config.name
        self.l1, self.l2 = config.l1, config.l2
        self.drop_rate, self.in_dim = config.dropout, config.in_dim
        frozen(config.seed)
        self.dropout = tc.nn.Dropout(config.dropout)
        self.linear1 = tc.nn.Linear(config.in_dim, 1)
        self.lossfn = tc.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, X):
        H = self.dropout(X.cuda())
        return self.linear1(H).flatten().to(X.device)

    def compute_loss(self, X, Y):
        R = self(X.cuda())
        Y = Y.squeeze().cuda()
        L = self.lossfn(tc.sigmoid(R), Y) +\
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
    config = LSRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LSRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
