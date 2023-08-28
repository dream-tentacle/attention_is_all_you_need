import torch
import torch.nn as nn
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, d_model, hidden) -> None:
        super().__init__()
        self.W_1 = nn.Linear(d_model, hidden)
        self.W_2 = nn.Linear(hidden, d_model)

    def forward(self, x) -> Tensor:
        """
        x (Tensor): (B,L,d_model)
        """
        return self.W_2(nn.functional.relu(self.W_1(x)))
