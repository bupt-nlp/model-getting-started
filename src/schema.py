from __future__ import annotations

from typing import List


class SequenceLabelingExample:
    def __init__(self, tokens: List[str], labels: List[str]):
        self.tokens = tokens
        self.labels = labels


class SentenceClassificationExample:
    """
    sentence classification example
    """
    def __init__(self, sentence: str, category: str):
        self.sentence = sentence
        self.category = category
