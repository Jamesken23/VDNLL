import math
import torch
import torch.nn as nn
import torch.optim as optim

"""
Self_Attention网络
Ref: https://github.com/goddoe/text-classification-and-sentence-representation/blob/master/sentence_representation/self_attn.py
"""

class SelfAttention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embedding_dim (int): dimension of embedding.
            padding_idx: 表示用于填充的参数索引, 比如用3填充,嵌入向量索引为3的向量设置为0
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes)
        )
        # self.fc = nn.Linear(self.embedding_dim, num_classes)
        
        self.optimizer = optim.Adam(self.parameters(), 0.01)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        """Feed-forward CNN.
        Args:
            X (torch.Tensor): inputs, shape of (batch_size, sequence).
        Returns:
            torch.tensor, Sentence representation.
        """
        batch_size, seq_len = X.size()
        X = self.embeddings(X) # batch size x seq_len x embed_dim
        
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X) # batch_size x seq_len
        
        attention_score_raw = q @ k.transpose(-2,-1) / math.sqrt(self.embedding_dim) # batch_size x seq_len x seq_len
        
        attention_score = torch.softmax(attention_score_raw, dim=2) # batch_size x seq_len x seq_len
        
        weighted_sum = attention_score @ v # batch_size x seq_len x dim

        context = torch.mean(weighted_sum, dim=1) # self attention된 임베딩들을 average
        # print("context shape is {0}".format(context.shape))
        return context, self.classifier(context)