from __future__ import annotations

import os
from typing import List
from tap import Tap


class ConfigBase(Tap):
    """use typed argument parser to capture arguments
    """
    random_seed: int = 50
    train: bool = True
    pretrained_model: str = "hfl/chinese-bert-wwm-ext"
    train_val_split_ratio: float = 0.7
    
    device :str = "cuda:1"
    pos_neg_example_num_ratio :str = 2  # =neg/pos

    # for training period
    model_type: str = "bert-sequence-classification"

    max_length: int = 128
    epoch: int = 7
    learning_rate: float = 0.0001
    batch_size: int = 16

    num_threshold: float = 0.7

    @property
    def bad_case_predict_path(self):
        return os.path.join(self.root_folder, "bad_case_predict")

    @property
    def root_folder(self):
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    @property
    def raw_data_folder(self):
        return os.path.join(self.root_folder, "labeled_data")
    
    @property
    def output_dir(self):
        """save the model to the output layer
        """
        return os.path.join(self.root_folder, 'output', self.model_type)


class Config:
    """create global singleton config container in order to parse arguments multi-times.
    """
    _instance: ConfigBase = None

    @classmethod
    def instance(cls) -> ConfigBase:
        if not cls._instance:
            # ignores extra arguments and only parses known arguments.
            cls._instance = ConfigBase().parse_args(known_only=True)
        return cls._instance

