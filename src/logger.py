from __future__ import annotations


from typing import Optional, Union, List

import torch
import numpy as np
from logging import (
    Logger,
)

from collections import defaultdict
from loguru import logger

from src.schema import MetricsReport, NumInputExample, RPTInputExample


def create_logger(log_file: Optional[str] = None) -> Logger:
    """
    create logger to record the things
    """
    if log_file:
        logger.add(log_file)
    return logger

def convert_to_numpy_data(data) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.cpu().detach().numpy()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    assert isinstance(data, np.ndarray)
    return data


def inference(model, dataset_loader, device:str="cuda:0") -> MetricsReport:

    y_pred_array = []
    y_true_array = []

    with torch.no_grad():

        for _, input_ids, token_type_ids, attention_mask, single_batch_labels in dataset_loader:

            input_ids, token_type_ids, attention_mask= input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)

            bert_out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            logits = bert_out.logits
            logits = torch.nn.functional.softmax(logits, dim=-1)
            y_pred_array += torch.argmax(logits, dim=-1).view(-1).tolist()
            y_true_array += single_batch_labels.view(-1).tolist()

    y_pred_array = np.array(y_pred_array)
    y_true_array = np.array(y_true_array)
    report = MetricsReport.by_sequence(y_pred_array, y_true_array)
    return report

def inference_report(model, dataset_loader, examples: List[Union[NumInputExample, RPTInputExample]], device):
    """inference on num label

    Args:
        model (pytorch Module subclass): the model
        dataloader ([type]): [description]
        device (str, optional): the device that model belong to. Defaults to "cuda:0".
    """
    if isinstance(model, str):
        model = torch.load(model, map_location=device)
    y_pred_array, y_true_array = [], []
    table_pred, table_truth = defaultdict(list), defaultdict(list)
    label_pred, label_truth = defaultdict(list), defaultdict(list)

    id2example = {example.id: example for example in examples}

    with torch.no_grad():

        for ids, input_ids, token_type_ids, attention_mask, single_batch_labels in dataset_loader:

            input_ids, token_type_ids, attention_mask= input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)

            bert_out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            logits = bert_out.logits
            logits = torch.nn.functional.softmax(logits, dim=-1)

            logit_index = torch.argmax(logits, dim=-1).view(-1).tolist()
            y_pred_array += logit_index

            truth_index = single_batch_labels.view(-1).tolist()
            y_true_array += truth_index

            ids = ids.detach().numpy()
            for index, id in enumerate(ids):
                example = id2example[id]

                table_pred[example.table_id].append(logit_index[index])
                table_truth[example.table_id].append(truth_index[index])

                label_pred[example.label].append(logit_index[index])
                label_truth[example.label].append(truth_index[index])


    y_pred_array = np.array(y_pred_array)
    y_true_array = np.array(y_true_array)


    pair_report = MetricsReport.by_sequence(y_pred_array, y_true_array)

    # construct table report
    table_result = []
    for key in table_pred.keys():
        table_result.append(all(np.array(table_pred[key]) == np.array(table_truth[key])))
    
    label_report = {}
    for label in label_pred.keys():
        label_report[label] = MetricsReport.by_sequence(label_pred[label], label_truth[label])
    
    table_report = MetricsReport.by_sequence(table_result, [True] * len(table_result))

    return pair_report, table_report, label_report
