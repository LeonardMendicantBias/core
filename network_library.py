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

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
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

        score = self._attention(q, k, mask)
        output = torch.matmul(score, v).contiguous().view(b, -1, self.d_model)

        return self.out(output), score

        
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_model*4)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(d_model*4, d_model)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class Layer(nn.Module):

    def __init__(self,
        d_model,
        mha_head,
        is_post_norm,
        is_cross, mhca_head,
        drop_prob=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.is_post_norm = is_post_norm
        self.is_cross = is_cross

        self.mha = MHA(d_model, mha_head)
        self.drop = nn.Dropout(drop_prob)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.ff_norm = nn.LayerNorm(d_model)

        if is_cross:
            self.mhca = MHA(d_model, mhca_head)
            self.mhca_norm = nn.LayerNorm(d_model)
            self.cff = FeedForward(d_model)
            self.cff_norm = nn.LayerNorm(d_model)

    def forward(self, x, x_e=None):
        x_ = x if self.is_post_norm else self.mha_norm(x)
        x, score = x + self.drop(self.mha(x_, x_, x_))
        x = self.mha_norm(x) if self.is_post_norm else x
            
        x_ = x if self.is_post_norm else self.ff_norm(x)
        x = x + self.ff(x_)
        x = self.ff_norm(x) if self.is_post_norm else x

        if x_e and self.is_cross:
            x_ = x if self.is_post_norm else self.mhca_norm(x)
            x, cscore = x + self.drop(self.mhca(x_, x_e, x_e))
            x = self.mhca_norm(x) if self.is_post_norm else x

            x_ = x if self.is_post_norm else self.cff_norm(x)
            x = x + self.cff(x_)
            x = self.cff_norm(x) if self.is_post_norm else x

            return x, score, cscore
        return x, score, None
        

# class TransformerEncoder(nn.Module):

#     def __init__(self, n_layers, d_model, n_heads, is_post_norm):
#         super().__init__()
#         self.layers = nn.ModuleList([Layer(d_model, n_heads, is_post_norm, False, 0) for _ in n_layers])
        
#     def forward(self):
#         scores = []
#         for layer in self.layers:
#             x, score, _ = layer(x)
#             scores.append(score.cpu().detach())
#         return x, scores


# class TransformerDecoder(nn.Module):

#     def __init__(self, n_layers, d_model, n_heads, is_post_norm, n_cheads):
#         super().__init__()
#         self.layers = nn.ModuleList([Layer(d_model, n_heads, is_post_norm, True, n_cheads) for _ in n_layers])
        
#     def forward(self):
#         scores, cscores = [], []
#         for layer in self.layers:
#             x, score, cscore = layer(x)
#             scores.append(score.cpu().detach())
#             cscores.append(cscore.cpu().detach())
#         return x, scores, cscores


class Transformer(nn.Module):

    def __init__(self,
        # vocab-related
        enc_vocab_len: int, enc_vocab_pad: int,
        dec_vocab_len: int, dec_vocab_pad: int,
        is_share_emb: bool,
        # mha-related
        d_model: int,
        enc_head: int, enc_layers: int,
        dec_head: int, dec_chead: int, dec_layers: int,
        drop_prob: int=0.1,
        # transformer-related
        is_post_norm: bool=True,  # according to original Transformer architecture
        is_enc_abs: bool=True, is_dec_abs: bool=True,
    ):
        super().__init__()
        
        if is_share_emb:
            self.enc_emb = self.dec_emb = nn.Embedding(enc_vocab_len, d_model, enc_vocab_pad)
        else:
            self.enc_emb = nn.Embedding(enc_vocab_len, d_model, enc_vocab_pad)
            self.dec_emb = nn.Embedding(dec_vocab_len, d_model, dec_vocab_pad)

        self.enc_drop = nn.Dropout(drop_prob)
        self.dec_drop = nn.Dropout(drop_prob)

        self.enc_pos = None if is_enc_abs else nn.Identity()
        self.dec_pos = None if is_dec_abs else nn.Identity()

        self.encoder = nn.ModuleList([
            Layer(d_model, enc_head, is_post_norm, False, -1) for _ in range(enc_layers)
        ])
        self.decoder = nn.ModuleList([
            Layer(d_model, dec_head, is_post_norm, True, dec_chead) for _ in range(dec_layers)
        ])

        self.linear = nn.Linear(d_model, dec_vocab_len)
        self.linear_norm = nn.Identity() if is_post_norm else nn.LayerNorm(d_model)
    
    def forward(self, inp_seq, out_seq):
        inp_seq = self.enc_pos(self.enc_emb(inp_seq))
        inp_seq = self.enc_drop(inp_seq)
        out_seq = self.enc_pos(self.enc_emb(out_seq))
        out_seq = self.dec_drop(out_seq)

        enc_out, enc_scores = self.encoder(inp_seq)
        dec_out, dec_scores, cross_scores = self.decoder(enc_out, out_seq)

        logits = self.linear(self.linear_norm(dec_out))

        return logits, enc_scores, dec_scores, cross_scores

if __name__ == '__main__':
    pass