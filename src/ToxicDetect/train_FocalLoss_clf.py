import os
import sys

import torch as tc
import numpy as np
import tqdm


from utils import frozen
from pipelines import train_CLF, TrainingConfig, load_datasets
from autoencoder import TopKSAE


class FLRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "FLRewarder")
def sigmoid_focal_loss(
    inputs: tc.Tensor,
    targets: tc.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> tc.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = tc.sigmoid(inputs)
    ce_loss = tc.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    #if alpha >= 0:
    #    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #    loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

class FLRewarder(tc.nn.Module):
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
        L = sigmoid_focal_loss(R, Y) +\
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
    config = FLRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = FLRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
