import torch
import torch.nn as nn
import torch.optim as optim

"""
LSTM网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
https://github.com/xashru/mixup-text/blob/master/models/text_lstm.py

首先，LSTM默认batch_first=False，即默认batch_size这个维度是在数据维度的中间的那个维度，即喂入的数据为【seq_len, batch_size, hidden_size】这样的格式。此时
lstm_out：【seq_len, batch_size, hidden_size * num_directions】
lstm_hn:【num_directions * num_layers, batch_size, hidden_size】

当设置batch_first=True时，喂入的数据就为【batch_size, seq_len, hidden_size】这样的格式。此时
lstm_out:【 batch_size, seq_len, hidden_size * num_directions】
lstm_hn:【num_directions * num_layers, batch_size, hidden_size】
"""

hidden_layers = 2


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5, bid=False):
        super(LSTM, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_hidden = embedding_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=hidden_layers, dropout=dropout, batch_first=True,
                            bidirectional=bid)

        # projection head
        self.g = nn.Sequential(
            nn.Linear(self.n_hidden * hidden_layers * (1 + bid), self.n_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, int(self.n_hidden / 2), bias=True)
        )

        # classifier head
        self.fc = nn.Linear(self.n_hidden * hidden_layers * (1 + bid), num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        self.optimizer = optim.Adam(self.parameters(), 0.001)


    def forward(self, x, perturbation=None, ignore_feat=False, forward_fc=True):
        x = self.embeddings(x)
        
        if perturbation is not None:
           x =  self.dropout(x)
           x += perturbation
        
        _, (x, _) = self.lstm(x)
        
        feature = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        projection = self.g(feature)

        if forward_fc:
            logits = self.fc(feature)
            if ignore_feat == True:
                return projection, logits
            else:
                return feature, projection, logits
        else:
            if ignore_feat == True:
                return projection
            else:
                return feature, projection
