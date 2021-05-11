"""data process tools"""
from __future__ import annotations

import csv
from typing import List, Literal
from src.schema import InputExample


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
