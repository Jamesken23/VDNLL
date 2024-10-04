import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from multi_head_attention import Multi_Head_Attention


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max_pool1d(x, kernel_size=x.shape[2])



class ConvMHSA_Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_heads=4, num_classes=2, dropout=0.5):
        super(ConvMHSA_Model, self).__init__()

        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.multi_head_attention = Multi_Head_Attention(self.embedding_dim, num_heads)
        self.conv_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=2)
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3)
        self.conv_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=4)

        self.GlobalMaxPool1d = GlobalMaxPool1d()

        self.fc1 = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.optimizer = optim.Adam(self.parameters(), 0.01)

    def forward(self, X):
        X = self.embeddings(X)

        X = self.multi_head_attention(X)
        X = X.permute(0, 2, 1).float()

        conv_1 = self.GlobalMaxPool1d(self.conv_1(X))
        conv_2 = self.GlobalMaxPool1d(self.conv_2(X))
        conv_3 = self.GlobalMaxPool1d(self.conv_3(X))
        conv = torch.cat([conv_1, conv_2, conv_3], dim=-1)
        conv = conv.view(conv.size(0), -1)
        out = F.relu(self.fc1(conv))
        out = self.dropout(out)

        return self.fc2(out)