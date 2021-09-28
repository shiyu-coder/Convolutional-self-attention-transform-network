from data.data_loader import ADDataset
from models.model import CSATNet
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import warnings

class Exp_model:
    def __init__(self, args):
        pass

    def _build_model(self):
        pass

    def _get_data(self, flag):
        pass

    def _select_optimizer(self):
        pass

    def _select_criterion(self):
        pass

    def train(self, setting):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def test(self, setting):
        pass





