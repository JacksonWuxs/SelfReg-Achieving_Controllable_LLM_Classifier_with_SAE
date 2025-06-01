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
    def __init__(self, sae, masks1, masks2, config):
        LinearRewarder.__init__(self, config) 
        self.sae = sae # (D, C)
        self.mask1 = tc.tensor(masks1).cuda() # (C)
        self.mask2 = tc.tensor(masks2).cuda() # (C)
        for param in self.sae.parameters():
            param.requires_grad = False
        self.penalty = 1.0 

    def decompose_features(self, H):
        # H: the original hidden state of LLM
        # X: non-sensitive features
        # Z: sensitive features
        A = tc.relu(self.sae._encode(H)) * self.mask1 * self.mask2
        X = H - self.sae._decode(A)
        return X, A

    def compute_constraint(self, Z, R):
        S = self.sae._encode(self.linear1.weight)
        l1 = tc.abs(S[0, self.mask1 == 1]).mean()
        l2 = tc.abs(S[1, self.mask2 == 1]).mean()
        return self.penalty * (l1 + l2)

        return self.penalty * (tc.abs(S[:, self.mask == 1])).mean()

    def compute_loss(self, H, Y):
        X, Z = self.decompose_features(H.cuda())
        R = self(X)
        D = self.compute_constraint(Z, R)
        Y = Y.squeeze().cuda().reshape(-1)
        L = tc.nn.CrossEntropyLoss(weight=tc.tensor([2.5, 2.5, 1]).cuda(),
                                   label_smoothing=0.0)(R, Y.long()) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters()) +\
            D
        P = R.argmax(axis=1).squeeze().reshape(-1)
        A = (P == Y).sum() / R.shape[0]
        return L, A, P.detach().cpu().tolist(), Y.detach().cpu().tolist()



if __name__ == "__main__":
    config = PTRewarderConfig()
    l = 16
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=l)
    sae = TopKSAE.from_disk("outputs/TopK7_l%d_h65k_FT_epoch40_DD.pth" % l)
    mask1, mask2 = choose_untask_mask("annotations/TopAct_TopK7_l%d_h65k_FT_epoch40_DD_DD_explained_UnP.tsv" % l,)
    model = SelfReg(sae, mask1, mask2, config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
