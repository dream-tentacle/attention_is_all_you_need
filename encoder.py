import torch
import torch.nn as nn
from attention import multi_head_attention
from FFN import feed_forward_network


class Transformer_encoder(nn.Module):
    def __init__(self, d_model, h) -> None:
        """
        a transformer encoder layer
        d_model / h = d_k = d_v
        Args:
            d_model (int): the embedding dimension
            h (int): the heads number
        """
        super().__init__()
        self.WO = nn.Linear(d_model, d_model, bias=False)
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.h = h
        self.norm1 = nn.LayerNorm(normalized_shape=[d_model])
        self.norm2 = nn.LayerNorm(normalized_shape=[d_model])
        self.FFN = feed_forward_network(d_model=d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B,L,d_model]
        """
        x = self.norm1(
            multi_head_attention(x, x, x, self.WO, self.WQ, self.WK, self.WV, self.h)
            + x
        )
        x = self.norm1(self.FFN(x) + x)
        return x
