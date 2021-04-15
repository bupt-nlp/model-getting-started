from __future__ import annotations

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score
)
from numpy import ndarray



@dataclass_json
@dataclass
class SentenceClassificationInputExample:
    id: int
    sentence: str
    category: str


@dataclass_json
@dataclass
class RPTInputExample:
    id:int
    header: str
    
    label: str
    label_description: str

    # 0: false, 1: true
    category: bool

    cell_value: str
    # we can 
    cell_type: str

    # this is for inference module
    row_index: int = -1
    column_index: int = -1
    table_id: int = -1

    def convert_to_bert_sentence_clssification_data(self) -> SentenceClassificationInputExample:
        return SentenceClassificationInputExample(
            sentence='[CLS]' + '[SEP]'.join([
                self.header,
                self.label, self.label_description
            ]),
            # TODO: will map the category
            category=str(int(self.category)),
            id=self.id
        )


@dataclass_json
@dataclass
class NumInputExample:
    id: int

    top_header: str
    left_header: str
    
    label: str
    label_description: str
    
    cell_value: str
    # we can 
    cell_type: str

    # 0: false, 1: true
    category: bool

    # this is for inference module
    row_index: int = -1
    column_index: int = -1
    table_id: int = -1

    def convert_to_bert_sentence_clssification_data(self) -> SentenceClassificationInputExample:
        return SentenceClassificationInputExample(
            sentence='[CLS]' + '[SEP]'.join([
                self.label, self.label_description,
                self.top_header, self.left_header,
            ]),
            # convert bool value to the integer value
            category=str(int(self.category)),
            id=self.id
        )


@dataclass_json
@dataclass
class MetricsReport:
    accuracy: float
    recall: float
    precision: float
    f1_score: float

    def __str__(self): 
        return f"accuracy<{self.accuracy}>, recall<{self.recall}>, precision<{self.precision}> f1_score<{self.f1_score}>"

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

        from src import utils
        
        predicted: ndarray = utils.convert_to_numpy_data(predicted)
        truth: ndarray = utils.convert_to_numpy_data(truth)
        return MetricsReport(
            accuracy=accuracy_score(predicted, truth),
            recall=recall_score(predicted, truth),
            precision=precision_score(predicted, truth),
            f1_score=f1_score(predicted, truth),
        )
