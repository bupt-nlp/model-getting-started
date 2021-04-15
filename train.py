#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from random import shuffle, seed
from typing import List, Tuple
import os
import torch.nn as nn
import torch
from copy import deepcopy
from transformers import BertTokenizer, AdamW

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import defaultdict

from tools.analysis_num import analysis_table_info
from src.config import Config
from src.utils import create_logger, inference, inference_report
from src.schema import (
    NumInputExample, SentenceClassificationInputExample,
)
from src.schema import MetricsReport
from src.model import get_model

from src.data_process import (
    read_data, 
    load_label_description, 
    load_num_input_examples, 
    load_data_loader_with_bert_tokenzier
)

from tqdm import tqdm

config = Config.instance()
logger = create_logger("train")

# 此配置暂时放在这里，因为不同模型的限定标签类型是不一样的，所以不适合放到Configuration里面
# label<FLUOT_NUM> has 19 pdf
# label<GENDERM_NUM> has 13 pdf
# label<LEVF_NUM> has 25 pdf
# label<LADIA_NUM> has 34 pdf
# label<PROCEDT_NUM> has 41 pdf
# label<PVI_NUM> has 23 pdf
# label<AGEMEAN_NUM> has 26 pdf
# label<RECUR12_NUM> has 5 pdf
# label<RFT_NUM> has 6 pdf
# label<AADPRE_NUM> has 8 pdf
# label<GENDERF_NUM> has 3 pdf

# valid_labels = ["FLUOT_NUM", "GENDERM_NUM", "LEVF_NUM", "LADIA_NUM", "PROCEDT_NUM", "PVI_NUM", "AGEMEAN_NUM", "RFT_NUM", "AADPRE_NUM"]
valid_labels = ["FLUOT_NUM", "GENDERM_NUM", "LEVF_NUM", "LADIA_NUM", "PROCEDT_NUM", "AGEMEAN_NUM"]

train_labels: List[str] = valid_labels  # label list for trianing period
validation_labels: List[str] = valid_labels    # labels list for validation for period
inference_labels: List[str] = valid_labels  # label list for inference period

def load_data(return_examples: bool = False):
    """load dataloader from input examples
    """
    train_data, validation_data = read_data()
    config = Config.instance()
    seed(config.random_seed)

    # 根据限定标签来筛选数据
    train_data = {label: data for label, data in train_data.items() if label in train_labels}
    validation_data = {label: data for label, data in validation_data.items() if label in validation_labels}
    inference_data = {label: data for label, data in validation_data.items() if label in inference_labels}

    # print the analysis info to the console
    analysis_table_info(train_data, validation_data, inference_data)
    
    # extract label description and build a dict for it
    label_description = load_label_description()

    # 重新构造正负例样本
    train_pos_examples, train_neg_examples = load_num_input_examples(
        train_data, label_description, language_index=1,
    )
    shuffle(train_neg_examples)
    train_neg_examples = train_neg_examples[:len(train_pos_examples) * config.pos_neg_example_num_ratio]
    train_data_examples = train_pos_examples + train_neg_examples
    train_data: List[SentenceClassificationInputExample] = [example.convert_to_bert_sentence_clssification_data() for example in train_data_examples]

    validation_pos_examples, validation_neg_examples = load_num_input_examples(
        validation_data, label_description, language_index=1
    )
    inference_data_examples = validation_pos_examples + validation_neg_examples

    shuffle(validation_neg_examples)
    validation_neg_examples = validation_neg_examples[:len(validation_pos_examples) * config.pos_neg_example_num_ratio]
    validation_data_examples = validation_pos_examples + validation_neg_examples
    validation_data: List[SentenceClassificationInputExample] = [example.convert_to_bert_sentence_clssification_data() for example in validation_data_examples]

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    train_data_loader, validation_data_loader = load_data_loader_with_bert_tokenzier(train_data, tokenizer), load_data_loader_with_bert_tokenzier(validation_data, tokenizer)
    inference_data = [example.convert_to_bert_sentence_clssification_data() for example in inference_data_examples]
    inference_data_loader = load_data_loader_with_bert_tokenzier(inference_data, tokenizer)
    if return_examples:
        return (train_data_loader, validation_data_loader, inference_data_loader), (train_data_examples, validation_data_examples, inference_data_examples)
    return train_data_loader, validation_data_loader, inference_data_loader


def train():

    # 1. get configuration & init output dir
    config = Config.instance()
    writer = SummaryWriter(log_dir=config.output_dir)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 2. load model & data

    device = torch.device(config.device)
    model = get_model(config.model_type, model_params={"pretrained_model_name_or_path": config.pretrained_model})

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    (train_data_loader, validation_data_loader, inference_data_loader), (train_data, validation_data, inference_data) = load_data(return_examples=True)

    # 3. train the model
    best_score = -1
    count = 0
    # for epoch in tqdm(range(config.epoch)):
    for epoch in tqdm(range(0)):
        for i, (_, input_ids, token_type_ids, attention_mask, label) in enumerate(train_data_loader):
            input_ids, token_type_ids, attention_mask, label = input_ids.to(
                device), token_type_ids.to(device), attention_mask.to(device), label.to(device)

            bert_out = model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=label)

            # with different train stratege
            bert_out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/train", bert_out.loss, count)
            count += 1

            if i % 50 == 0:
                # loss.backward()
                report: MetricsReport = inference(model, validation_data_loader, device)

                logger.info(f'epoch {epoch}, {i}-th batch, loss<{bert_out.loss}>, accuracy<{report.accuracy}>, recall<{report.recall}>, precision<{report.precision}> f1_score<{report.f1_score}>')
                if best_score < report.f1_score:
                    model_path = os.path.join(config.output_dir, 'best_model.pt')
                    torch.save(model, model_path)
                    logger.success(f'save best model<f1_score<{best_score}> -> <{report.f1_score}>> to {model_path}')
                    best_score = report.f1_score
                
                model_path = os.path.join(config.output_dir, 'last_model.pt')
                torch.save(model, model_path)
                logger.success(f'save last model<<{report.f1_score}>> to {model_path}')
                
    writer.close()
    
    # inference on the pair and table module
    pair_report, table_report, label_report = inference_report(os.path.join(config.output_dir, 'best_model.pt'), inference_data_loader, inference_data, device)

    logger.info('==============REPORT BEGIN==============')

    logger.info(f"inference tables report : {table_report}")
    logger.info(f"inference pair report : {pair_report}")

    logger.info('==============LABEL REPORT BEGIN==============')
    for key, report in label_report.items():
        logger.info(f"label<{key}>   {report}")
    logger.info('==============LABEL REPORT END==============')
    

if __name__ == '__main__':
    train()