import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torch.optim as optim

"""
TextCNN+Transformer网络
参考：
https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/Transformer.py
https://github.com/CLOVEXCWZ/Pytorch_LongText_Classification_Demo/blob/master/models/transformer.py
https://github.com/Lizhen0628/text_classification/blob/master/model/models.py
"""

class Transformer_Conv(nn.Module):
    def __init__(self, vocab_size, seq_len,
                 n_class=2,
                 embed_dim=300,  # embedding的维度
                 dim_model=300,
                 dropout=0.5,
                 num_head=5,
                 hidden=1024,
                 num_encoder=2, device=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # -----------------------TextCNN---------------------------
        self.n_filters = 200
        
        self.conv = nn.Conv1d(in_channels=seq_len, out_channels=self.n_filters, kernel_size=5, padding=2)

        # -----------------------TextCNN---------------------------
        self.num_head = num_head
        
        self.postion_embedding = Positional_Encoding(self.n_filters, embed_dim, dropout, device)
        self.encoder = Encoder(self.n_filters, num_head, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_encoder)])
        
        self.fc1 = nn.Linear(self.n_filters*embed_dim, self.n_filters)
        self.fc2 = nn.Linear(self.n_filters, n_class)
        self.dropout = nn.Dropout(dropout)
        
        self.optimizer = optim.Adam(self.parameters(), 0.01)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        # embedded = embedded.permute(0, 2, 1).float()
        # embedded = [batch size, emb dim, sent len]

        conved = F.relu(self.conv(embedded)) # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # conved = F.max_pool1d(conved, int(conved.shape[2]))
        pooled = self.dropout(conved) # pooled_n = [batch size, n_filters]

        # cat = self.dropout(torch.cat(pooled, dim=1)) # cat = [batch size, n_filters * len(filter_sizes)]
#         print("pooled shape is {0}".format(pooled.shape))
        out = pooled.reshape(-1, pooled.shape[2], pooled.shape[1]) # [batch size, n_filters, filter_sizes]
                
        out = self.postion_embedding(out) # [batch size, n_filters, filter_sizes]

        for encoder in self.encoders:
            out = encoder(out) # [batch size, n_filters, filter_sizes]

        out = out.view(out.size(0), -1) # [batch size, n_filters * filter_sizes]
        # out = torch.mean(out, 1)
        out = self.dropout(self.fc1(out))
        
        return self.fc2(out)


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale

        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out