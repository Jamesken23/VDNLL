import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

"""
BiLSTM+Attention网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
"""


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device, num_classes=2):
        super(BiLSTM_Attention, self).__init__()
        self.n_hidden = embedding_dim
        self.bid = True
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, self.n_hidden, bidirectional=self.bid)
        
        self.out = nn.Linear(self.n_hidden * (1 + self.bid), num_classes)
        
        self.optimizer = optim.Adam(self.parameters(), 0.01)
        self.criterion = nn.CrossEntropyLoss()

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        hidden = final_state.view(-1, self.n_hidden * (1 + self.bid), 1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]

        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]

        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        if torch.cuda.is_available():
            return context, soft_attn_weights.data.cpu().numpy()
        else:
            return context, soft_attn_weights.data.numpy()

    def forward(self, text):
        input = self.embedding(text)   # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1 * 2, len(text), self.n_hidden)).to(self.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1 * 2, len(text), self.n_hidden)).to(self.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))

        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]

        attn_output, attention = self.attention_net(output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return attn_output, self.out(attn_output) 


if __name__ == '__main__':
    # 1. 自己构造一批数据
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    # 2. 数据预处理
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    vocab2id = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(vocab2id)

    inputs = []
    for sen in sentences:
        inputs.append(np.asarray([vocab2id[n] for n in sen.split()]))

    # 3. 超参数
    embedding_dim = 2   # 词嵌入维度
    n_hidden = 5   # 隐层维度
    num_classes = 2  # 0 or 1

    targets = []
    for out in labels:
        targets.append(out)  # To using Torch Softmax Loss function

    input_batch = Variable(torch.LongTensor(inputs))
    target_batch = Variable(torch.LongTensor(targets))

    model = BiLSTM_Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output, attention = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()