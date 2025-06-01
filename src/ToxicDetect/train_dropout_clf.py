import os
import sys

import torch as tc
import numpy as np
import tqdm


from pipelines import (train_CLF, TrainingConfig, load_datasets,
                       construct_mask, choose_untask_mask, post_check)
from autoencoder import TopKSAE
from train_simple_clf import LinearRewarder


class DropoutRewarderConfig(TrainingConfig):
    def __init__(self):
        TrainingConfig.__init__(self, "DropoutRewarder")
        self.dropout = 0.1



if __name__ == "__main__":
    config = DropoutRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda()
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=True)
