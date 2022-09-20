# %%

import math
from pyexpat import model
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class APE(nn.Module):

    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:,:seq_len]* math.sqrt(self.d_model)
    

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

    def post_norm(self, x, x_e):
        x_ = x

        x, score = self.mha(x, x, x)
        x = x_ + x

        x = self.mha_norm(x)

    def pre_norm(self, x, x_e):
        x_ = x

        x = self.mha_norm(x)
        x, score = self.mha(x, x, x)

        x = x + x_
        

    def forward(self, x, x_e=None):
        res_x = x
        x = x if self.is_post_norm else self.mha_norm(x)
        x, score = self.mha(x, x, x)
        x = res_x + self.drop(x)
        x = self.mha_norm(x) if self.is_post_norm else x

        res_x = x 
        x = x if self.is_post_norm else self.ff_norm(x)
        x = res_x + self.ff(x)
        x = self.ff_norm(x) if self.is_post_norm else x

        if x_e is not None and self.is_cross:
            res_x = x
            x = x if self.is_post_norm else self.mhca_norm(x)
            x, cscore = self.mhca(x, x_e, x_e)
            x = res_x + self.drop(x)
            x = self.mhca_norm(x) if self.is_post_norm else x

            res_x = x
            x = x if self.is_post_norm else self.cff_norm(x)
            x = res_x + self.cff(x)
            x = self.cff_norm(x) if self.is_post_norm else x

            return x, score, cscore
        return x, score, None
        

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
        t = q.shape[1]
        b, s, _ = k.shape

        k = self.key(k).view(b, s, self.num_heads, self.d_head).transpose(1,2)
        q = self.query(q).view(b, t, self.num_heads, self.d_head).transpose(1,2)
        v = self.value(v).view(b, s, self.num_heads, self.d_head).transpose(1,2)

        score = self._attention(q, k, mask)
        output = torch.matmul(score, v).contiguous().view(b, -1, self.d_model)

        return self.out(output), score


class Transformer(nn.Module):

    def __init__(self,
        # vocab-related
        name: str,
        enc_vocab_len: int, enc_pad_idx: int,
        dec_vocab_len: int, dec_pad_idx: int,
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
        assert (a:=d_model % enc_head) == 0, f"head dim of enc is not appropriate"
        assert (a:=d_model % dec_chead) == 0, f"head dim of dec is not appropriate"

        super().__init__()
        self.name = name
        self.d_model = d_model
        if is_share_emb:
            self.enc_emb = self.dec_emb = nn.Embedding(enc_vocab_len, d_model, padding_idx=enc_pad_idx)
        else:
            self.enc_emb = nn.Embedding(enc_vocab_len, d_model, padding_idx=enc_pad_idx)
            self.dec_emb = nn.Embedding(dec_vocab_len, d_model, padding_idx=dec_pad_idx)

        self.enc_drop = nn.Dropout(drop_prob)
        self.dec_drop = nn.Dropout(drop_prob)

        self.enc_pos = APE(d_model) if is_enc_abs else nn.Identity()
        self.dec_pos = APE(d_model) if is_dec_abs else nn.Identity()

        self.encoder = nn.ModuleList([
            Layer(d_model, enc_head, is_post_norm, False, -1) for _ in range(enc_layers)
        ])
        self.decoder = nn.ModuleList([
            Layer(d_model, dec_head, is_post_norm, True, dec_chead) for _ in range(dec_layers)
        ])

        self.linear = nn.Linear(d_model, dec_vocab_len)
        self.linear_norm = nn.Identity() if is_post_norm else nn.LayerNorm(d_model)
    
    def forward(self, inp_seq, out_seq):
        inp_seq = self.enc_pos(self.enc_emb(inp_seq)*math.sqrt(self.d_model))
        inp_seq = self.enc_drop(inp_seq)
        out_seq = self.dec_pos(self.dec_emb(out_seq)*math.sqrt(self.d_model))
        out_seq = self.dec_drop(out_seq)

        inp_scores = []
        for layer in self.encoder:
            inp_seq, score, _ = layer(inp_seq)
            inp_scores.append(score)

        out_scores, out_cscores = [], []
        for layer in self.decoder:
            out_seq, score, cscore = layer(out_seq, inp_seq)
            out_scores.append(score)
            out_cscores.append(cscore)

        logits = self.linear(self.linear_norm(out_seq))

        return logits, inp_scores, out_scores, out_cscores

if __name__ == '__main__':
    pass