import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from bilstm import BiLSTM
from bigru import BiGRU


class scvd_sa(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len, num_classes=2, dropout=0.5):
        super(scvd_sa, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_filters = 200

        self.bilstm = BiLSTM(vocab_size, embedding_dim)
        self.bigru = BiGRU(vocab_size, embedding_dim)

        self.conv_1 = nn.Conv1d(in_channels=seq_len, out_channels=embedding_dim, kernel_size=2)
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3)
        self.conv_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=4)

        # self.fc11 = nn.Linear(embedding_dim * 250, self.n_filters)
        # self.fc12 = nn.Linear(self.n_filters, 1)
        self.fc12 = nn.Linear(embedding_dim * 250, 1)

        # self.fc21 = nn.Linear(embedding_dim, self.n_filters)
        # self.fc22 = nn.Linear(self.n_filters, 1)
        self.fc22 = nn.Linear(embedding_dim, 1)
        # self.fc31 = nn.Linear(embedding_dim, self.n_filters)
        # self.fc32 = nn.Linear(self.n_filters, 1)
        self.fc32 = nn.Linear(embedding_dim, 1)

        # self.fc41 = nn.Linear(self.n_filters * 3, embedding_dim)
        # self.fc41 = nn.Linear(self.n_filters + embedding_dim*2, embedding_dim)
        self.fc41 = nn.Linear(embedding_dim * 250 + embedding_dim * 2, embedding_dim)
        self.fc42 = nn.Linear(embedding_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, text, proj=False):
        x = self.embedding(text)

        conved_1 = F.relu(self.conv_1(x))
        # conved = F.max_pool1d(conved, int(conved.shape[2]))
        pooled_1 = self.dropout(conved_1)
        conved_2 = F.relu(self.conv_2(pooled_1))
        # conved = F.max_pool1d(conved, int(conved.shape[2]))
        pooled_2 = self.dropout(conved_2)
        conved_3 = F.relu(self.conv_3(pooled_2))
        conv = conved_3.view(conved_3.size(0), -1)

        bilstm_vec, _ = self.bilstm(text, proj=True)
        bigru_vec, _ = self.bigru(text, proj=True)

        # print("conv shape is:", conv.shape)
        # vec_1 = self.relu(self.fc11(conv))
        # weight_1 = self.sigmoid(self.fc12(vec_1))
        # # 点积操作
        # new_vec_1 = torch.mul(vec_1, weight_1)

        weight_1 = self.sigmoid(conv)
        # 点积操作
        new_vec_1 = torch.mul(conv, weight_1)

        # vec_2 = self.relu(self.fc21(bilstm_vec))
        # weight_2 = self.sigmoid(self.fc22(vec_2))
        # new_vec_2 = torch.mul(vec_2, weight_2)

        weight_2 = self.sigmoid(self.fc22(bilstm_vec))
        new_vec_2 = torch.mul(bilstm_vec, weight_2)

        # vec_3 = self.relu(self.fc31(bigru_vec))
        # weight_3 = self.sigmoid(self.fc32(vec_3))
        # new_vec_3 = torch.mul(vec_3, weight_3)

        weight_3 = self.sigmoid(self.fc32(bigru_vec))
        new_vec_3 = torch.mul(bigru_vec, weight_3)

        # print("new_vec_1 shape is {0}, new_vec_2 shape is {1}, and new_vec_3 shape is {2}".format(new_vec_1.shape, new_vec_2.shape, new_vec_3.shape))
        # 合并张量
        mergevec = torch.cat([new_vec_1, new_vec_2, new_vec_3], dim=1)
        flattenvec = mergevec.view(mergevec.size(0), -1)
        # print("flattenvec shape is {0}".format(flattenvec.shape))
        finalmergevec = self.dropout(self.relu(self.fc41(flattenvec)))
        prediction = self.fc42(finalmergevec)

        if proj is True:
            return finalmergevec, prediction
        else:
            return prediction