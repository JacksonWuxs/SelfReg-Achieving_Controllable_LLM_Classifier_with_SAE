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
        self.penalty = 3.0

    def decompose_features(self, H):
        # H: the original hidden state of LLM
        # X: non-sensitive features
        # Z: sensitive features
        A = tc.relu(self.sae._encode(H)) * self.mask
        X = H - self.sae._decode(A)
        return X, A

    def compute_constraint(self, Z, R):
        S = self.sae._encode(self.linear1.weight)
        return self.penalty * (tc.abs(S[:, self.mask == 1])).mean()

    def compute_loss(self, H1, H2):
        assert H1.shape == H2.shape
        X1, Z1 = self.decompose_features(H1.cuda())
        X2, Z2 = self.decompose_features(H2.cuda())
        R1, R2 = self(X1.cuda()), self(X2.cuda())
        Z, R = tc.vstack([Z1, Z2]), tc.hstack([R1, R2])
        Y = tc.tensor([1] * R1.shape[0] + [0.] * R2.shape[0]).cuda().squeeze()
        D = self.compute_constraint(Z, R)
        L = self.lossfn(R, Y) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters()) +\
            D
        A = (R1 > R2).sum() / R1.shape[0]
        P = tc.hstack([tc.where(R1 > R2, 1., 0.),
                       tc.where(R2 > R1, 1., 0.)])
        return L, A, P.cpu().numpy(), Y.cpu().numpy()



if __name__ == "__main__":
    config = SelfRegConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    sae = TopKSAE.from_disk("../FinetuneSAE/outputs/TopK7_l16_h65k_v3FT_epoch5_RM.pth")
    mask = choose_untask_mask("annotations/TopAct_TopK7_l16_h65k_FT_epoch5_RM_RM_explained_TaskMiniReward.tsv")
    model = SelfReg(sae, mask, config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
