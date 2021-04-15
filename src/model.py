from __future__ import annotations

import torch
from transformers import BertTokenizer

from src.config import Config
from transformers import BertForSequenceClassification


class BertSequenceClassificationModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.bert_model = BertForSequenceClassification.from_pretrained(
            **kwargs, num_labels = 2
        )

    def forward(self, input_ids, token_type_ids, attention_mask, labels = None):
        output = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask= attention_mask,
            labels=labels
        )
        return output


class ModelContainer:

    _num_model: torch.Module = None
    _rpt_model: torch.Module = None
    _bert_tokenzier = None
    
    @classmethod
    def init(cls):
        config = Config.instance()
        # this is for servering configuration, which is fixed
        cls._num_model = torch.load('/home/human/wjj/TableUnderstand/output/bert-sequence-classification/best_model.pt')
        cls._rpt_model = torch.load('/home/human/qzq/TableUnderstandingExperiment/output/bert-sequence-classification/best_model.pt')
        cls._num_model.to('cpu')
        cls._rpt_model.to('cpu')
        cls._bert_tokenzier = BertTokenizer.from_pretrained(config.pretrained_model)
        
    @classmethod
    def num_model(cls):
        """get the num-model
        """
        if not cls._num_model:
            raise ValueError(f"please init global singleton instance model")
        return cls._num_model

    @classmethod
    def tokenizer(cls):
        if not cls._bert_tokenzier:
            raise ValueError(f"please init global singleton instance bert tokenzier")
        return cls._bert_tokenzier


# this mapping will work for different solution for
models = {
    "bert-sequence-classification": BertSequenceClassificationModel,
}


def get_model(name, model_params={}):
    """different models containes different key-word arguments
    """
    if name not in models:
        raise ValueError(f"{name} not in model hub")
    model = models[name](**model_params)
    return model
