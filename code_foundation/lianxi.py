import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    x_sum = np.sum(x, axis=axis, keepdims=True)

    return e_x / x_sum

def numpy_attention(x, wq, wk, wv, mask=None):
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = q @ k.swapaxes(-2, -1) / np.sqrt(k.shape[-1])
    if mask is not None:
        scores = np.where(mask == 0, scores, -1e9)

    attn = softmax(scores)
    out = attn @ v

    return out

class SelfAttention(nn.Module):
    def __init__(self, d_model):

        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.outproj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.outproj(out)

        return out