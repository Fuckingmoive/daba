import torch
import torch.nn as nn
import torch.nn.functional as F


# 这个代码中省略了输入矩阵x转变为qkv的过程！！！

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        # 定义K, Q, V的权重矩阵
        # 多头注意力中K、Q、V的线性层具有相同输入和输出尺寸是一种常见且实用的设计选择！！！
        self.k_linear = nn.Linear(d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # 分头后的维度
        self.d_token = d_model // heads
        # 定义输出权重矩阵
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        # 计算batch大小
        batch = q.size(0)

        # 线性变换后的Q, K, V，然后分割成多个头
        k = self.k_linear(k).view(batch, -1, self.heads, self.d_token)
        q = self.q_linear(q).view(batch, -1, self.heads, self.d_token)
        v = self.v_linear(v).view(batch, -1, self.heads, self.d_token)

        # 转置调整维度，以计算注意力分数
        k = k.transpose(1, 2)  # 形状变为 [batch, heads, seq_len, d_token]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算自注意力分数
        scores = self.attention(q, k, v, self.d_token)

        # 调整形状以进行拼接
        scores = scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        # 通过输出权重矩阵进行线性变换
        output = self.out(scores)
        return output

    @staticmethod
    def attention(q, k, v, d_token):
        # 计算注意力分数 (q @ k^T) / sqrt(d_token)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_token)
        # 应用softmax归一化（沿着最后一个维度（dim=-1））
        attn = F.softmax(scores, dim=-1)
        # 计算加权的V
        output = torch.matmul(attn, v)
        return output


# import torch.nn as nn
# import torch
# import math
# import torch.nn.functional as F
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads, d_model, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_model // heads  # 512 / 8
#         self.h = heads
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(d_model, d_model)
#
#     def attention(self, q, k, v, d_k, mask=None, dropout=None):
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # self-attention公式
#         # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#             scores = scores.masked_fill(mask == 0, -1e9)
#         scores = F.softmax(scores, dim=-1)  # self-attention公式
#         if dropout is not None:
#             scores = dropout(scores)
#         output = torch.matmul(scores, v)  # self-attention公式
#         return output
#
#     def forward(self, q, k, v, mask=None):
#         bs = q.size(0)  # 进行线性操作划分为成 h 个头
#         k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
#         q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
#         v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
#         # 矩阵转置
#         k = k.transpose(1, 2)
#         q = q.transpose(1, 2)
#         v = v.transpose(1, 2)  # 计算 attention
#         scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
#         # 连接多个头并输入到最后的线性层
#         concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
#         output = self.out(concat)
#         return output
#
#
# # 准备q、k、v张量
# d_model = 512
# num_heads = 8
# batch_size = 32
# seq_len = 64
#
# q = torch.randn(batch_size, seq_len, d_model)  # 64 x 512
# k = torch.randn(batch_size, seq_len, d_model)  # 64 x 512
# v = torch.randn(batch_size, seq_len, d_model)  # 64 x 512
#
# sa = MultiHeadAttention(heads=num_heads, d_model=d_model)
# print(sa(q, k, v).shape)  # torch.Size([32, 64, 512])
# print('')