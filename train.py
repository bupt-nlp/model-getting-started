from __future__ import annotations

from src.models.base_model import SequenceLabelingModel
from src.schema import Config


def create_sequence_labeling_model() -> SequenceLabelingModel:
    pass

def train():
    config = Config.parse_args(known_only=True)

    # 1. create model
    model = create_sequence_labeling_model()

    # 2. load data
    dataset = None

    # 3. train 
    # 3.1 warmup<learning_rate_scheduler>

    
    


if __name__ == '__main__':
    train()