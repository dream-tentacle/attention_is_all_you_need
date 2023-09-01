import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from attention import MultiheadAttention
from FFN import FFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model, head, hidden_size, dropout=0.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.attn = MultiheadAttention(d_model, head, dropout)
        self.ffn = FFN(d_model, hidden=hidden_size, dropout=dropout)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x, mask):
        x = self.norm1(self.attn(x, x, x, mask) + x)
        x = self.norm2(self.ffn(x) + x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer_num, d_model, head, hidden_size, dropout=0.5) -> None:
        super().__init__()
        self.encoders = clone(
            EncoderLayer(d_model, head, hidden_size, dropout), layer_num
        )

    def forward(self, x, mask):
        for i in range(len(self.encoders)):
            # print(x.device)
            x = self.encoders[i](x, mask)
        return x
