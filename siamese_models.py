import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm


from tap import Tap


class Config(Tap):
    learning_rate: float = 0.001
    do_learn: bool = True
    batch_size: int = 64
    epochs: int = 6
    weight_decay: float = 0.001

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 7)
        self.conv2 = torch.nn.Conv2d(64, 128, 5)
        self.conv3 = torch.nn.Conv2d(128, 256, 5)

        self.max_pooler = torch.nn.MaxPool2d(2)

        self.output = torch.nn.Sequential(
            nn.Linear(2304, 512),
            nn.Linear(512, 2)
        )
        
    def forward(self, data):
        