import os
import sys

import torch as tc
import numpy as np
import tqdm


from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE
from train_simple_clf import LinearRewarder


class LSRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "FLRewarder")
        self.gamma = 2.




class FocalLoss(tc.nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, outputs, targets):
        ce_loss = tc.nn.functional.cross_entropy(outputs, targets, reduction='none', 
                weight=tc.tensor([2.5, 2.5, 1.0]).cuda())
        pt = tc.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss


class FLRewarder(LinearRewarder):
    def __init__(self, config):
        LinearRewarder.__init__(self, config)
        self.lossfn = FocalLoss(gamma=config.gamma)

    def compute_loss(self, X, Y):
        R = self(X.cuda())
        Y = Y.squeeze().cuda()
        L = self.lossfn(R, Y.long()) +\
            self.l1 * sum(tc.norm(_, p=1) for _ in self.parameters()) +\
            self.l2 * sum(tc.norm(_, p=2) for _ in self.parameters())
        P = R.argmax(axis=1).squeeze()
        A = (P == Y).sum() / R.shape[0]
        return L, A, P.detach().cpu().tolist(), Y.detach().cpu().tolist()


if __name__ == "__main__":
    config = LSRewarderConfig()
    print(config.gamma)
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = FLRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config)
