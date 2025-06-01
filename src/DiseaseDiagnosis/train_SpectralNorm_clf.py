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
        TrainingConfig.__init__(self, "SN2Rewarder")



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


class SNRewarder(LinearRewarder):
    def __init__(self, config):
        LinearRewarder.__init__(self, config)

    def compute_loss(self, X, Y):
        R = self(X.cuda())
        Y = Y.squeeze().cuda()
        L = tc.nn.CrossEntropyLoss(weight=tc.tensor([2.5, 2.5, 1.]).cuda(),
                                   label_smoothing=self.label_smooth)(R, Y.long()) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters()) +\
            0.001 * compute_spectral_norm(self.linear1.weight, training=self.training)
        P = R.argmax(axis=1).squeeze()
        A = (P == Y).sum() / R.shape[0]
        return L, A, P.detach().cpu().tolist(), Y.detach().cpu().tolist()


if __name__ == "__main__":
    config = LinearRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = SNRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
