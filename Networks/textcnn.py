import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

"""
TextCNN网络, TextCNN_1d
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter, filter_sizes, output_dim=2, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.optimizer = optim.Adam(self.parameters(), 0.01)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch size, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个conved

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch,num_filter]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filter * len(filter_sizes)]
        # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。

        return cat, self.fc(cat)


if __name__ == "__main__":
    vocab_size = 150  # 词典数量
    dmodel = 256  # embedding层词向量

    num_filter = 100  # 卷积核个数
    filter_size = [2, 3, 4]  # 卷积核的长，取了三种
    output_dim = 2  # 种类

    model = TextCNN(vocab_size, dmodel, num_filter=num_filter, filter_sizes=filter_size, output_dim=output_dim).to(device)
    # optimizer = optim.Adam(model.parameters(), 0.01)
    # criterion = nn.CrossEntropyLoss().to(device)
