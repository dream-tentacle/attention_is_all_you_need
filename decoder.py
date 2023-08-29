import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from attention import MultiheadAttention
from FFN import FFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model, head, hidden_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.attn = MultiheadAttention(d_model, head)
        self.attn2 = MultiheadAttention(d_model, head)
        self.ffn = FFN(d_model, hidden=hidden_size)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.norm3 = nn.LayerNorm([d_model])

    def forward(self, x, enc_memory, mask):
        x = self.norm1(self.attn(x, x, x, mask) + x)
        x = self.norm2(self.attn2(x, enc_memory, enc_memory) + x)
        x = self.norm3(self.ffn(x) + x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer_num, d_model, head, hidden_size) -> None:
        super().__init__()
        self.encoders = clone(EncoderLayer(d_model, head, hidden_size), layer_num)

    def forward(self, x, enc_memory, mask=None):
        for i in range(len(self.encoders)):
            x = self.encoders[i](x, enc_memory, mask)
        return x
