# %%

import math
from pyexpat import model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_len = 80):
#         super().__init__()
#         self.d_model = d_model
        
#         # create constant 'pe' matrix with values dependant on 
#         # pos and i
#         pe = torch.zeros(max_seq_len, d_model)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = \
#                 math.sin(pos / (10000 ** ((2 * i)/d_model)))
#                 pe[pos, i + 1] = \
#                 math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         # make embeddings relatively larger
#         x = x * math.sqrt(self.d_model)
#         #add constant to embedding
#         seq_len = x.size(1)
#         x = x + self.pe[:,:seq_len]
#         return x


class MHA(nn.Module):

    def __init__(self,
        d_model, num_heads,
        dropout_prob
    ):
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(d_model, d_model)

    def _attention(self, q, k, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_head)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        return F.softmax(scores, dim=-1)

    def forward(self, q, k, v, mask=None):
        t = q.shape[0]
        b, s, *_ = k.shape

        k = self.key(k).view(b, -1, self.num_heads, self.d_head).transpose(1,2)
        q = self.query(q).view(b, -1, self.num_heads, self.d_head).transpose(1,2)
        v = self.value(v).view(b, -1, self.num_heads, self.d_head).transpose(1,2)

        scores = self._attention(q, k, mask)
        output = torch.matmul(scores, v).contiguous().view(b, -1, self.d_model)

        return self.out(output)

        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

