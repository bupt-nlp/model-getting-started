from __future__ import annotations

from src.data_process import AgNewsDataProcessor

def test_agnews_data_processor():
    """test agnews data processors
    """
    processor = AgNewsDataProcessor()
    assert len(processor.get_labels()) == 4

    examples = processor.get_train_examples('./data/agnews/train.csv')
    assert len(examples) > 0