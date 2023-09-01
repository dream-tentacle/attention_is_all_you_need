import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def multy(X: torch.Tensor, Y: torch.Tensor):
    if len(X.shape) == 2:
        return X.matmul(Y)
    return X.bmm(Y)


def attention(K, Q, V, mask=None):
    weights = multy(Q, K.transpose(-1, -2))
    if mask is not None:
        weights = weights.masked_fill(mask=mask, value=-1e9)
    weights = F.softmax(weights, dim=-1) / (K.shape[-1] ** 0.5)
    return multy(weights, V)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, head, dropout=0.5) -> None:
        super().__init__()
        self.linears = clone(nn.Linear(d_model, d_model, bias=False), 4)
        self.head = head
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [B,L,d_model]
            k: [B,L,d_model]
            v: [B,L,d_model]
            mask (optional): [B,L,L]. Defaults to None.
        Returns:
            Tensor: [B,L,d_model]
        """
        k = self.linears[0](k)
        q = self.linears[1](q)
        v = self.linears[2](v)
        length = int(self.d_model / self.head)

        def calcu_head(i):
            return attention(
                k[..., i * length : (i + 1) * length],
                q[..., i * length : (i + 1) * length],
                v[..., i * length : (i + 1) * length],
                mask=mask,
            )

        att = torch.concat([calcu_head(i) for i in range(self.head)], dim=-1)
        return self.dropout(self.linears[3](att))
