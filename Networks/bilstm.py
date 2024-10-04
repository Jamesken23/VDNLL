import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
BiLSTM网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
https://github.com/xashru/mixup-text/blob/master/models/text_lstm.py
"""

hidden_layers = 2


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5, bid=True):
        super(BiLSTM, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_hidden = embedding_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=hidden_layers,
                            dropout=dropout,
                            bidirectional=bid)
        self.fc1 = nn.Linear(self.n_hidden * hidden_layers * (1 + bid), embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

        self.optimizer = optim.Adam(self.parameters(), 0.01)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, proj=False):
        x = self.embeddings(x)
        x = x.permute(1, 0, 2)
        #         print("x shape is:", x.shape)
        _, (x, _) = self.lstm(x)

        pred = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        pred = self.fc1(pred)
        out = self.fc2(pred)

        if proj is True:
            return pred, out
        else:
            return out