import os
import sys

import torch as tc
import numpy as np
import tqdm


from pipelines import (train_CLF, TrainingConfig, load_datasets,
                       construct_mask, choose_untask_mask, post_check)
from autoencoder import TopKSAE
from train_simple_clf import LinearRewarder 


class SelfRegConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "SelfReg")


class SelfReg(LinearRewarder):
    def __init__(self, sae, masks, config):
        LinearRewarder.__init__(self, config) 
        self.sae = sae # (D, C)
        self.checks = [i for i, m in enumerate(masks) if m == 1]
        self.mask = tc.tensor(masks).cuda() # (C)
        for param in self.sae.parameters():
            param.requires_grad = False
        self.penalty = 31.0

    def decompose_features(self, H):
        # H: the original hidden state of LLM
        # X: non-sensitive features
        # Z: sensitive features
        A = tc.relu(self.sae._encode(H)) * (self.mask == 1)
        X = H - self.sae._decode(A)
        return X, A

    def compute_constraint(self, Z, R):
        S = self.sae._encode(self.linear1.weight)
        return self.penalty * (tc.abs(S[:, self.mask == 1])).mean()

    def compute_loss(self, H, Y):
        Y = Y.cuda().squeeze(1)
        X, Z = self.decompose_features(H.cuda())
        R = self(X)
        D = self.compute_constraint(Z, R)
        L = self.lossfn(R, Y) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters()) +\
            D
        P = tc.where(R > 0., 1.0, 0.0)
        A = (P == Y).sum() / R.shape[0]
        return L, A, P.detach().cpu().tolist(), Y.detach().cpu().tolist()



if __name__ == "__main__":
    config = SelfRegConfig()
    l = 16
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=l)
    sae = TopKSAE.from_disk("../FinetuneSAE/outputs/TopK7_l%d_h65k_FT_epoch5_TD.pth" % l)
    mask = choose_untask_mask("annotations/TopAct_TopK7_l%d_h65k_FT_epoch5_TD_explained_TaskHarmful.tsv" % l,)
    model = SelfReg(sae, mask, config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
