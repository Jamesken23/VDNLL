
import torch
import torch.nn as nn
import torch.optim as optim

"""
GRU网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
https://github.com/xashru/mixup-text/blob/master/models/text_lstm.py
https://github.com/Lizhen0628/text_classification/blob/master/model/models.py
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layers = 2


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5, bid=True):
        super(GRU, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_hidden = embedding_dim

        # LSTM layer
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=hidden_layers, dropout=dropout,
                            bidirectional=bid)
        self.fc1 = nn.Linear(self.n_hidden * hidden_layers * (1 + bid), self.n_hidden // 2)
        self.fc2 = nn.Linear(self.n_hidden // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # self.optimizer = optim.Adam(self.parameters(), 0.01)
        self.optimizer = optim.SGD(self.parameters(), 0.01)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(1, 0, 2)

        _, x = self.gru(x)
        
        x = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        x_1 = self.fc1(x)
        x_2 = self.fc2(self.dropout(x_1))

        return x_1, x_2
#         return x_2