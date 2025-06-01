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
        TrainingConfig.__init__(self, "L2normRewarder")
        self.l2 = 0.0001



if __name__ == "__main__":
    config = LinearRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config, loading_only=False)
