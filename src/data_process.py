"""data process tools"""
from __future__ import annotations

import csv
import os
from typing import List, Dict
# from typing import Literal
import torch

from src.schema import InputExample
from src.schema import InputFeatures
import jieba


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AgNewsDataProcessor(DataProcessor):
    """
    process the agnews
    Args:
        DataProcessor ([type]): [description]
    """
    def get_labels(self):
        return [1, 2, 3, 4]

    def get_examples(self, file: str) -> List[InputExample]:
        lines = self._read_tsv(file)

        examples: List[InputExample] = []
        for index, (label, title, description) in enumerate(lines[1:]):
            example = InputExample(
                guid=f'guid-{index}',
                text_a=title,
                text_b=description,
                label=label
            )
            examples.append(example)
        
        return examples

    def get_train_examples(self, data_dir) -> List[InputExample]:
        return self.get_examples(data_dir)
    
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        return self.get_examples(data_dir)
    
    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir)


class THCNewsDataProcessor(DataProcessor):
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = {}

    def get_labels(self):
        return range(9)

    def get_examples(self, file: str) -> List[InputExample]:
        lines = self._read_txt(file, encoding='utf-8')

        examples: List[InputExample] = []

        for index, line in enumerate(lines):
            title, label = line.split('\t')
            label = (int)(label)

            example = InputExample(
                guid=f'guid-{index}',
                text_a=title,
                text_b=None,
                label=label
            )
            examples.append(example)

        return examples

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        return self.get_examples(data_dir)

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """
        get evaluation examples which is eval in training period
        """
        return self.get_examples(data_dir)

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir)

    def _read_txt(self, file, encoding='utf-8'):
        with open(file, 'r', encoding=encoding) as f:
            return f.readlines()


class THCNewsFeatures(InputFeatures):

    def get_train_features(self, vocab, data_dir):
        return self.get_features(vocab, self.get_examples(data_dir))

    def get_dev_features(self, vocab, data_dir):
        return self.get_features(vocab, self.get_examples(data_dir))

    def get_test_features(self, vocab, data_dir):
        return self.get_features(vocab, self.get_examples(data_dir))

    def get_features(self, vocab, examples: List[InputExample], language='zh') -> List[InputFeatures]:
        """Transform text to chat id for Chinese text."""
        features: List[InputFeatures] = []
        for _, example in enumerate(examples):
            if language == 'zh':
                input_ids = [vocab[char] if char in vocab.keys() else vocab['<UNK>']
                         for char in example.text_a]
            else:
                input_ids = [vocab[char] if char in vocab.keys() else vocab['<UNK>']
                             for char in example.text_a.split()]
            label_id = example.label
            feature = InputFeatures(
                input_ids=input_ids,
                attention_mask=None,
                segment_ids=None,
                label_id=label_id,
                is_real_example=False
            )
            features.append(feature)
        return features


class DataIterator():
    """
    A iterator can get batches from dataset
    """
    def __init__(self, features: List[InputFeatures],
                 batch_size: int,
                 max_len: int,
                 padding_fill: int):
        self.features = features
        self.batch_size = batch_size
        self.max_len = max_len
        self.padding_fill = padding_fill

    def _get_inputs(self, max_len: int, do_test=False) -> (
            torch.LongTensor, torch.LongTensor):
        """A generator returning input matrix for train, eval and test"""
        """
        @parameters:
        do_test: bool, If do_test is True, Y will be set as None
        max_len: int, the max size of inputs text
        """
        # set input matrix x, shape [features size, max sequence length]
        x = torch.LongTensor(len(self.features), max_len)
        # set input label matrix, shape [features size, 1]
        y = torch.LongTensor(len(self.features), 1)
        # if length of inputs is more than max_len, clip inputs, else pad them.
        for index, feature in enumerate(self.features):
            input_ids = feature.input_ids
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                padding = [self.padding_fill for _ in range(max_len - len(input_ids))]
                input_ids.extend(padding)
            x[index] = torch.LongTensor(input_ids)
            y[index] = feature.label_id
        if do_test:
            y = None
        return x, y

    def get_batch(self, batch_size, do_test: bool=False) -> (
            torch.LongTensor, torch.LongTensor):
        """Return a iterator, the item of iterator shape is [batch_size, max_len]"""
        x, y = self._get_inputs(max_len=self.max_len)
        batches = len(x) // batch_size
        if batches * batch_size < len(x):
            batches += 1
        # generate batch data
        for batch in range(batches):
            batch_x = x[batch*batch_size:(batch+1)*batch_size]
            batch_y = y[batch*batch_size:(batch+1)*batch_size]

            if len(batch_x) < batch_size:
                batch_x = torch.LongTensor(batch_size, self.max_len)
                batch_x[:len(x)-batch*batch_size] = x[batch*batch_size:]
                batch_y = torch.LongTensor(batch_size, 1)
                batch_y[:len(x)-batch*batch_size] = y[batch * batch_size:]

            if do_test:
                y = None

            yield batch_x, batch_y


if __name__ == '__main__':
    # # test for agnews
    # data_processor = AgNewsDataProcessor()
    # root = os.path.join(os.path.abspath('..'), 'data')
    # agnews = os.path.join(root, 'agnews')
    # train_file = os.path.join(agnews, 'train.csv')
    # train_examples = data_processor.get_examples(train_file)
    # test_file = os.path.join(agnews, 'test.csv')
    # test_examples = data_processor.get_examples(test_file)
    # print(f'Trainset Length: {len(train_examples)}, Example: {train_examples[0]}')
    # print(f'Testset Length: {len(test_examples)}, Example: {test_examples[0]}')

    # # test for thcnews
    data_processor = THCNewsDataProcessor()
    root = os.path.join(os.path.abspath('..'), 'data')
    agnews = os.path.join(root, 'THCNews')
    train_file = os.path.join(agnews, 'train.txt')
    train_examples = data_processor.get_examples(train_file)
    dev_file = os.path.join(agnews, 'dev.txt')
    dev_examples = data_processor.get_examples(dev_file)
    test_file = os.path.join(agnews, 'test.txt')
    test_examples = data_processor.get_examples(test_file)
    print(f'Trainset Length: {len(train_examples)}, Example: {train_examples[0]}')
    print(f'Dev Length: {len(dev_examples)}, Example: {dev_examples[0]}')
    print(f'Testset Length: {len(test_examples)}, Example: {test_examples[0]}')

    # Dataset Info:
    def text_statistic(examples):
        max_len = -1
        average_len = -1
        min_len = 99999
        sum = 0
        for example in examples:
            sen_len = len(example.text_a)
            max_len = sen_len if sen_len > max_len else max_len
            min_len = sen_len if sen_len < min_len else min_len
            sum += sen_len
        print(f'Max length: {max_len}, Min length: {min_len}, Average length: {sum/len(examples)}')
    
    text_statistic(train_examples)
    text_statistic(dev_examples)
    text_statistic(test_examples)