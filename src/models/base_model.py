from __future__ import annotations

import torch.nn as nn


class SequenceLabelingModel(nn.Module):
    def __init__(self):
        super().__init__()


class SequenceClassificationModel(nn.Module):
    """A base class for sequence classification task"""
    pass


class BertSequenceClassificationModel(SequenceClassificationModel):
    """A model sequence classification task which encoding by BERT"""
    pass


class BiLSTMSequenceClassificationModel(SequenceClassificationModel):
    """Base LSTM implement a model for classification which encoding by w2v"""
    def __init__(self, config):
        super(BiLSTMSequenceClassificationModel, self).__init__()

        if config.pretrained_model is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.pretrained_model, freeze=False
            )
        else:
            self.embedding = nn.Embedding(config.n_vocab,
                                          config.embedding_dim,
                                          padding_idx=config.n_vocab-1
                                        )
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)
        self._init_weights()

    def _init_weights(self):
        for name, w in self.named_parameters():
            if 'embedding' not in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def forward(self, input):
        x = self.embedding(input)  # [batch_size, seq_len, embedding_dim]
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
        output = self.fc(x[:, -1, :])  # [batch_size, num_classes]

        return output