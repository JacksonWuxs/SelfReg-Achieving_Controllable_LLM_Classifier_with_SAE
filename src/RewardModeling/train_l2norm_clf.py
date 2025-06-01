import os
import sys

import torch as tc
import numpy as np
import tqdm


from pipelines import (train_CLF, TrainingConfig, load_datasets,
                       construct_mask, choose_untask_mask, post_check)
from autoencoder import TopKSAE
from train_simple_clf import LinearRewarder


class L2NormRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "L2NormRewarder")
        self.l2 = 1e-3



if __name__ == "__main__":
    config = L2NormRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda()
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=True)
