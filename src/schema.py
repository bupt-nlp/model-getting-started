from __future__ import annotations

from typing import List, Dict, Optional

from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
import numpy as np
from tap import Tap
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    # TP TN FP FN
    confusion_matrix
)


class Config(Tap):
    data_dir: str  # The input data dir. Should contain the .tsv files (or other data files) for the task.
    task_name: str = 'Sequence-Classification'
    model_name: str = 'bilstm-for-sequence-classification'
    vocab_file: str
    output_dir: str  # save model and result
    log_dir: str  # save log
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    pretrained_path: str
    pretrained_model_name: str = 'random'  # the pretrained model name
    pretrained_model = None  # pretrained model or matrix
    n_vocab: int  # the size of vocab, set value in running
    embedding_dim: int  # the dim of embedding vector
    hidden_size: int  # hidden size for lstm
    num_layers: int = 2# num of lstm layer
    num_labels: int = 2  # The default is a 2-classes model
    dropout: int = 0.5

    do_lower_case: bool  # do lower case
    max_seq_length: int  # max sequence length
    do_train: bool
    do_eval: bool
    do_predict: bool

    train_batch_size: int = 128
    eval_batch_size: int = 64
    predict_batch_size: int = 8

    learning_rate: float = 1e-3
    epochs = 8
    require_improve = 1000
    warmup_proportion: float = 0.1
    save_checkpoints_steps: int = 100
    language: str = 'zh'


    @staticmethod
    def instance(cls) -> Config:
        """use single instance pattern for configuration

        Returns:
            Config: the Instance of Configuration
        """
        return ConfigInstanceCache.instance()


# class Config(Tap):
#     data_dir: str  # The input data dir. Should contain the .tsv files (or other data files) for the task.
#     task_name: str = 'Sequence-Classification'
#     model_name: str = 'bilstm-for-sequence-classification'
#     vocab_file: str
#     output_dir: str
#
#     pretrained_model_name: str = ''  # the pretrained model name
#     do_lower_case: bool  # do lower case
#     max_seq_length: int  # max sequence length
#     do_train: bool
#     do_eval: bool
#     do_predict: bool
#     train_batch_size: int = 32
#     eval_batch_size: int = 8
#     predict_batch_size: int = 8
#     learning_rate: float = 5e-5
#     epochs = 8
#     warmup_proportion: float = 0.1
#     save_checkpoints_steps: int = 100
#     language: str = 'zh'
#
#     num_labels: int = 2
#
#
#     @staticmethod
#     def instance(cls) -> Config:
#         """use single instance pattern for configuration
#
#         Returns:
#             Config: the Instance of Configuration
#         """
#         return ConfigInstanceCache.instance()

class ConfigInstanceCache:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance:
            return cls._instance
        cls._instance = Config.parse_args(known_only=True)
        return cls._instance


class GlobalData:
    def __init__(self):
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}


global_data = GlobalData()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self) -> str:
        des = f"InputExample: ID<{self.guid}> Label<{self.label}> Text-A<{self.text_a}>"
        if self.text_b:
            des += f" Text-B{self.text_b}"
        return des


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

    def get_feature_from_example(self, examples: InputExample) -> None:
        pass
    
    def __str__(self) -> str:
        return f"input_ids<{self.input_ids}>\nattention_mask<{self.attention_mask}>\nsegment_ids<{self.segment_ids}>\nlabel_id<{self.label_id}>"


@dataclass_json
@dataclass
class MetricsReport:
    # TODO: TP, FP -> 
    accuracy: float
    recall: float
    precision: float
    f1_score: float
    confusion_matrix: np.ndarray

    def __str__(self): 
        return "accuracy<{:.4f}>, recall<{:.4f}>, precision<{:.4f}> f1_score<{:.4f}>".format(
            self.accuracy, self.recall, self.precision, self.f1_score
        )

    def detail_report(self): 
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return "accuracy<{:.4f}>, recall<{:.4f}>, precision<{:.4f}> f1_score<{:.4f}> confusion_matrix<TN<{:.1f}>, FP<{:.1f}>, FN<{:.1f}>, TP<{:.1f}>>".format(
            self.accuracy, self.recall, self.precision, self.f1_score, tn, fp, fn, tp
        )

    @staticmethod
    def by_sequence(predicted, truth) -> MetricsReport:
        """generate metric score by predicted and truth
        Args:
            predicted ([type]): predicted value
            truth ([type]): truth value
        Returns:
            MetricsReport: the final MetricsReport
        """
        # 1. check the type of the object
        
        # convert data to numpy data type
        # predicted: np.ndarray = Util.convert_to_numpy_data(predicted)
        # truth: npndarray = Util.convert_to_numpy_data(truth)
        
        return MetricsReport(
            accuracy=accuracy_score(truth, predicted),
            recall=recall_score(truth, predicted),
            precision=precision_score(truth, predicted),
            f1_score=f1_score(truth, predicted),
            confusion_matrix=confusion_matrix(truth, predicted, labels=[0, 1])
        )

