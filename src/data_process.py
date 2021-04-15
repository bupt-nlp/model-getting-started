from __future__ import annotations

import os, json, random, math
from typing import List
from collections import defaultdic

import torch
from transformers import (
    BertTokenizer
)
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.utils.dummy_pt_objects import BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST

from src.schema import (
    NumInputExample,
    RPTInputExample,
    SentenceClassificationInputExample,
    MetricsReport
)
from src.config import Config

### 
# gloabl methods
###

def load_label_description():
    """get label-description mapping data
    """
    config = Config.instance()

    det_examples_file = os.path.join(config.raw_data_folder,"DET_Examples.xlsx")
    det_examples_data = pd.read_excel(det_examples_file, engine='openpyxl')

    labels = det_examples_data['标签'].tolist()
    label_description = det_examples_data['标签描述'].tolist()

    label_description = {label: str(describe) for label, describe in zip(labels, label_description)}
    return label_description

def get_cell_value_type(cell_value: str) -> str:
    """return the type of cell value which may influence the result of the model
    """
    # TODO: use spacy & jieba to detect the cell value type
    return cell_value

def _extract_labels(table_labels):
    # get all labels which are involving in that table
    labels = []
    n, m = len(table_labels), len(table_labels[0])
    for i in range(n):
        for j in range(m):
            if table_labels[i][j] != '0':
                eles = table_labels[i][j].split('[JOIN]')
                # print(eles)
                for ele in eles:
                    tmp = ele.split('##')
                    # print(tmp)
                    labels.append(tmp[0])

    return list(set(labels))

def read_data(seed=0, train=True):
    """ hhh, this method return 1 result when train = false, but return 2 result when train = True
    :param raw_file_folder:
    :param split_ratio: for a label, default 80% as train
    :return:
    """
    # 数据固定路径
    config = Config.instance()
    split_ratio = config.train_val_split_ratio

    collected_data = defaultdict(list)
    table = None

    for file in os.listdir(os.path.join(config.root_folder, 'data')):
        if not file.endswith('.json'):
            continue
        try:
            file_path = os.path.join(config.root_folder, 'data', file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for i in range(len(data['data'])):

                    if 'label' not in data['data'][i]:
                        continue

                    matched_labels = _extract_labels(data['data'][i]['label']) 
                    labels = data['data'][i]['label']
                    table = data['data'][i]['raw_table']
                    if train == False:
                        collected_data['data'].append((table, labels))
                    else:
                        for label in matched_labels:
                            collected_data[label].append((table, labels))
        except Exception as e:
            print(file)
            raise

    if train == False:
        return collected_data['data']

    # 基于label划分
    train_data, test_data = {}, {}
    for label, tables in collected_data.items():

        group_index = list(range(len(tables)))

        random.Random(seed).shuffle(group_index)

        slice_index = int(math.ceil(len(tables) * split_ratio))

        train_data[label] = [tables[i] for i in group_index[:slice_index]]
        test_data[label] = [tables[i] for i in group_index[slice_index:]]

    return train_data, test_data


### 
# NUM related methods
### 

def load_num_input_examples(
        label_tables,
        label_description_map: dict,
        language_index: int = 1,
        need_both_header: bool = True,
        label_list: List[str] = [''],
    ) -> List[NumInputExample]:
    """construct the num example data

    Args:
        label_tables ([type]): the target num label data
        label_descriptions ([type], optional): the description for the label. Defaults to None.
        language_index (int, optional): 0: chinese, 1: english. Defaults to 1.
        all_pos_data: bool -> 是否所有的数据都是正例数据
        need_both_header: bool -> 是否所有的header都是非空值
        label_list: List[str] -> 构造所有其他类别的负例样本

    Returns:
        List[NumInputExample]: the result of the NumInputExample
    """
    pos, neg = [], []
    # define the unique key for the example
    identifier = 0
    table_id = 0
    for key, tables in label_tables.items():
        
        if '_NUM' not in key:
            continue

        # whether I need to add label description in the preprocessing

        label_description = ""
        if key in label_description_map:
            label_description = label_description_map[key]
            if '/' in label_description:
                label_description = label_description.split('/')[language_index]

        for raw_table, labels in tables:
            table_id += 1
            # get the pos & net examples
            for row_index, row in enumerate(raw_table):
                if row_index == 0:
                    continue

                for column_index, cell in enumerate(row):
                    if column_index == 0:
                        continue

                    cell_value = str(cell).replace('<s>', '').replace('</s>', '')

                    # TODO: data enhancement
                    # 1. cross table training data (label -> label-list)
                    # 2. len(pos): len(neg) = 1: 1

                    input_example = NumInputExample(
                        top_header=raw_table[0][column_index],
                        left_header=raw_table[row_index][0],
                        label=key,
                        label_description=label_description, 
                        cell_value = cell_value,
                        cell_type=get_cell_value_type(cell),
                        category=0, 

                        row_index=row_index,
                        column_index=column_index,

                        id=identifier,
                        table_id=table_id
                    )

                    # both header is empty
                    if not input_example.top_header and not input_example.left_header:
                        continue

                    if need_both_header and (not input_example.top_header or not input_example.left_header):
                        continue
                    
                    identifier += 1

                    label = labels[row_index][column_index]

                    if '_NUM' in label:
                        input_example.category = 1
                        pos.append(input_example)
                    
                    # 构造负例样本
                    for neg_label in label_list:
                        if neg_label == key: continue
                        neg.append(NumInputExample(
                            top_header=raw_table[0][column_index],
                            left_header=raw_table[row_index][0],

                            label=key,
                            label_description=label_description, 
                            cell_value = cell_value,
                            cell_type=get_cell_value_type(cell),
                            category=0, 

                            row_index=row_index,
                            column_index=column_index,

                            id=identifier,
                            table_id=table_id
                        ))
                        identifier += 1
    return pos, neg

def load_data_loader_with_bert_tokenzier(input_examples: List[SentenceClassificationInputExample], tokenizer: BertTokenizer, sample: bool = True) -> DataLoader:
    """batch encode sentences to the input features

    Args:
        sentences (List[str]): the SentenceClassificationInputExample
        tokenizer (BertTokenizer): the bert-based tokenzier
        sample (bool): whether to sample data from examples
    """
    sentences = [example.sentence for example in input_examples]

    # this is the unifier key for example
    ids = [example.id for example in input_examples]

    # the value list: ['0', '1']
    categories = [int(example.category) for example in input_examples]

    config = Config.instance()
    features = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=config.max_length,
        return_tensors='pt'
    )

    data = TensorDataset(
        torch.IntTensor(ids),
        features['input_ids'], 
        features['token_type_ids'], 
        features['attention_mask'],
        torch.LongTensor(categories),
    )

    # add sampler to the data_loader
    sampler = RandomSampler(data) if sample else None

    data_loader = DataLoader(data, sampler=sampler, batch_size=config.batch_size)
    return data_loader

### 
# RPT related methods
### 

def load_rpt_input_examples(
        label_tables,
        label_description_map: dict,
        language_index: int = 1,
        all_pos_data: bool = False
    ) -> List[RPTInputExample]:
    """construct the rpt example data

    Args:
        label_tables ([type]): the target num label data
        label_descriptions ([type], optional): the description for the label. Defaults to None.
        language_index (int, optional): 0: chinese, 1: english. Defaults to 1.

    Returns:
        List[RPTInputExample]: the result of the RPTInputExample
    """
    pos, neg = [], []
    
    # define the unique key for the example
    identifier = 0
    table_id = 0
    for key, tables in label_tables.items():
        
        if '_RPT' not in key:
            continue

        # whether I need to add label description in the preprocessing

        label_description = ""
        if key in label_description_map:
            label_description = label_description_map[key]
            if '/' in label_description:
                label_description = label_description.split('/')[language_index]

        for raw_table, labels in tables:
            
            table_id += 1
            # get the pos & net examples
            m1, n1 = len(raw_table), len(raw_table[0])
            m2, n2 = len(labels), len(labels[0])
            for row_index, row in enumerate(raw_table):

                for column_index, cell in enumerate(row):
                    valid_example_flag = False # RPT need only top or left header
                    cell_value = str(cell).replace('<s>', '').replace('</s>', '')

                    # TODO: data enhancement
                    # 1. cross table training data (label -> label-list)
                    # 2. len(pos): len(neg) = 1: 1
                    header = ""
                    if row_index == 0:
                        # RPT need only top or left header
                        header = raw_table[0][column_index]
                        valid_example_flag = True
                    elif column_index == 0:
                        # RPT need only top or left header
                        header = raw_table[row_index][0]
                        valid_example_flag = True
                    else:
                        pass
                    if header == "":
                        continue # we don't need empty row/column header and also here we skip cell region
                    input_example = RPTInputExample(
                            header=header,
                            label=key,
                            label_description=label_description, 
                            cell_value = cell_value,
                            cell_type=get_cell_value_type(cell),
                            category=0, 

                            row_index=row_index,
                            column_index=column_index,

                            id=identifier,
                            table_id=table_id
                        )
                    label = labels[row_index][column_index]
                    if valid_example_flag:
                        identifier += 1
                        if '_RPT' in label or all_pos_data:
                            input_example.category = 1
                            pos.append(input_example)
                        else:
                            neg.append(input_example)
    
    data = pos + neg
    return data
