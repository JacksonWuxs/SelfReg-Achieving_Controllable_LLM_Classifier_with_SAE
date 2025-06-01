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
        TrainingConfig.__init__(self, "LSRewarder")
        self.ls = 0.1



if __name__ == "__main__":
    config = LSRewarderConfig()
    train_data, valid_data, test1_data, test2_data = load_datasets(config.seed, layer=16)
    model = LinearRewarder(config).cuda() 
    train_CLF(train_data, valid_data, test1_data, test2_data, model, config)
