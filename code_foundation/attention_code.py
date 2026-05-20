import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Self_attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # scores: [batch_size, seq_len, seq_len]
        # 需要把 k 的最后两个维度转置
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 最后一维表示当前 token 对所有 key token 的分数。
        # 所以 softmax 要在最后一维做，使每个 query 对所有 key 的注意力权重和为 1。
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        return out
    
class Multi_head_attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # [B, S, D] -> [B, S, H, d_head] -> [B, H, S, d_head]
        q = self.w_q(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        # scores: [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_head))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # attn: [B, H, S, S]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # [B, H, S, S] 矩阵乘 [B, H, S, d_head] -> [B, H, S, d_head]
        # 实际上是[S, S] 乘以 [S, d_head]，得到[S, d_head]
        out = torch.matmul(attn, v)
        # out: [B, H, S, d_head] -> [B, S, H, d_head] -> [B, S, D]
        # contiguous() 是为了确保内存连续，view() 需要内存连续的张量
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.out_proj(out)

        return out


"""
numpy版本的attention
"""
def softmax(x):
    """
    np.max参数介绍：
    - axis: 指定沿哪个轴进行最大值计算。默认为None，表示计算整个数组的最大值。这里的-1表示沿最后一个轴计算最大值。
    以二维数组为例，如果axis=0，则计算每列的最大值；如果axis=1，则计算每行的最大值。axis=-1表示沿最后一个轴计算最大值，对于二维数组来说相当于axis=1。
    - keepdims: 是否保持原数组的维度。默认为False，表示返回的结果会丢失被计算轴的维度。如果设置为True，则会保留被计算轴的维度，结果中该轴的大小为1。
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention_numpy(x, wq, wk, wv, mask=None):
    """
    Self-Attention 核心函数
    Args: X(L,d) 输入序列, W_q/W_k/W_v(d,d) 投影矩阵
    Returns: output(L,d) 注意力输出
    """
    """
    np.dot 和 @ 运算符的区别：
    - np.dot(a, b) 是 NumPy 中的函数，用于计算两个数组的点积（dot product）。对于二维数组，np.dot(a, b) 等价于矩阵乘法。
    - @ 是 Python 3.5 引入的矩阵乘法运算符，可以用于 NumPy 数组和其他支持矩阵乘法的对象。对于二维数组，a @ b 等价于 np.dot(a, b)。
    在这个函数中，x @ wq 和 np.dot(x, wq) 是等价的，都表示对输入 x 进行线性变换得到查询向量 q。同样的，x @ wk 和 x @ wv 也是等价的。
    """
    q = x @ wq
    k = x @ wk
    v = x @ wv

    dk = q.shape[-1]
    scores = q @ k.T / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask == 0, scores, -1e9)
    attn = softmax(scores)
    out = attn @ v
    return out, attn

import numpy as np

def softmax(x):
    """
    实现 Softmax 函数。
    为了防止指数爆炸导致数值溢出（数值稳定性），在计算前减去最后一个维度上的最大值。
    """
    # 保持维度以进行正确的广播 (broadcasting)
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算缩放点积注意力
    参数:
        Q: Query 矩阵，shape 为 (..., seq_len_q, d_k)
        K: Key 矩阵，shape 为 (..., seq_len_k, d_k)
        V: Value 矩阵，shape 为 (..., seq_len_k, d_v)
        mask: 掩码矩阵 (可选)，用于遮蔽不需要计算注意力的位置
    返回:
        output: 注意力输出
        attention_weights: 注意力权重矩阵
    """
    # 提取 K 的特征维度大小
    d_k = K.shape[-1]
    
    # 1. 计算 Q 和 K 的点积注意力分数
    # K 需要在最后两个维度上转置。使用 swapaxes 能够完美兼容带 batch 和不带 batch 的情况
    # 分数 shape: (..., seq_len_q, seq_len_k)
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    
    # 2. 应用 Mask（如果存在）
    # 将 mask 对应为 0 的位置替换为极小的负数（如 -1e9），这样 Softmax 后这些位置的值就会趋近于 0
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    # 3. 计算 Softmax 得到注意力权重分布
    attention_weights = softmax(scores)
    
    # 4. 将权重与 V 相乘，得到加权求和后的表示
    # output shape: (..., seq_len_q, d_v)
    output = attention_weights @ V
    
    return output, attention_weights

class SelfAttentionLayer:
    """
    完整的自注意力层，包含 Q, K, V 的线性映射权重
    """
    def __init__(self, d_model, d_k, d_v):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        # 初始化权重矩阵 W_q, W_k, W_v
        # 实际深度学习框架中通常使用 Xavier 或 Kaiming 初始化，这里用随机标准差做简单演示
        self.W_q = np.random.randn(d_model, d_k) * 0.01
        self.W_k = np.random.randn(d_model, d_k) * 0.01
        self.W_v = np.random.randn(d_model, d_v) * 0.01

    def forward(self, X, mask=None):
        """
        前向传播
        参数:
            X: 输入序列，shape 为 (batch_size, seq_len, d_model)
        """
        # 线性映射生成 Q, K, V
        # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        # 计算注意力
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, attention_weights
