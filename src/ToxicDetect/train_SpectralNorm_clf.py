import os
import sys

import torch as tc
import numpy as np
import tqdm
from torch.nn.utils import spectral_norm

from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE



def reshape_weight_to_matrix(weight_mat, axis=0):
    if axis != 0:
        weight_mat = weight_mat.permute(
            axis, *[d for d in range(weight_mat.dim()) if d != axis]
            )
    return weight_mat.reshape(weight_mat.size(0), -1)


def compute_spectral_norm(W, axis=0, eps=1e-12, training=True):
    if not training:
        return 0.0
    W = reshape_weight_to_matrix(W, axis)
    h, w = W.size()
    u = tc.nn.functional.normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=eps)
    v = tc.nn.functional.normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=eps)
    with tc.no_grad():
        for _ in range(10):
            v = tc.nn.functional.normalize(
                tc.mv(W.t(), u), dim=0, eps=eps, out=v
                )
            u = tc.nn.functional.normalize(
                tc.mv(W, v), dim=0, eps=eps, out=u
                )
    return tc.dot(u, tc.mv(W, v))

class SNRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "SN2Rewarder")


class SNRewarder(tc.nn.Module):
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
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters()) +\
            0.02 * compute_spectral_norm(self.linear1.weight, training=self.training)
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
    config = SNRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = SNRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
