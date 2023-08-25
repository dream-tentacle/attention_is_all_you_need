import torch
from torch import Tensor
import torch.nn as nn


def matrix_multi(X: Tensor, Y: Tensor):
    if X.dim() == 3:
        return torch.bmm(X, Y)
    elif X.dim() == 2:
        return torch.matmul(X, Y)
    print(f"wrong dimension input in multiply : {X.shape} with {Y.shape}")
    return None


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    """
    B = batch size
    L = sequence length
    d_k, d_v as the paper shows

    Args:
        Q (Tensor): [B,L,d_k] or [L,d_k]
        K (Tensor): [B,L,d_k] or [L,d_k]
        V (Tensor): [B,L,d_v] or [L,d_v]

    Returns:
        softmax(Q*K.T/sqrt(d_k))*V
        Tensor: [B,d_v] or [d_v]
    """
    weights = torch.nn.functional.softmax(
        matrix_multi(Q, K.transpose(-1, -2)), dim=-2
    )  # [B,L,L]
    attention = matrix_multi(weights, V)  # [B,L,d_v]
    return attention


def multi_head_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    WO: nn.Linear,
    WQ: nn.Linear,
    WK: nn.Linear,
    WV: nn.Linear,
    h: int,
) -> Tensor:
    """
    WQ is the list of W_i^Q, WK and WV the same.

    Args:
        Q (Tensor): [B,L,d_model] or [L,d_model]
        K (Tensor): [B,L,d_model] or [L,d_model]
        V (Tensor): [B,L,d_model] or [L,d_model]
        WO (nn.Linear): [h*d_v,d_model]
        WQ (list[nn.Linear]): [h*d_k,d_model]
        WK (list[nn.Linear]): [h*d_k,d_model]
        WV (list[nn.Linear]): [h*d_v,d_model]
        h (int): the head number, d_k * h = d_model
    Returns:
        Tensor: [B,L,d_model]
    """
    Q2 = WQ(Q)  # [B,L,d_model]
    K2 = WK(K)
    V2 = WV(V)
    d_k = int(Q.shape[-1] / h)

    def head(i):
        return scaled_dot_product_attention(
            Q2[..., i * d_k : (i + 1) * d_k],  # [B,L,d_k]
            K2[..., i * d_k : (i + 1) * d_k],  # [B,L,d_k]
            V2[..., i * d_k : (i + 1) * d_k],  # [B,L,d_k]
        )  # [B,L,d_v]

    all_heads = torch.concat([head(i) for i in range(h)], dim=-1)  # [B,L,h*d_v]
    multi_head = WO(all_heads)  # [B,L,d_model]
    return multi_head  # [B,L,d_model]
