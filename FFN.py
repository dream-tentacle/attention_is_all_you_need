import torch
import torch.nn as nn
from torch import Tensor


class feed_forward_network(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)

    def forward(self, x) -> Tensor:
        """compute feed forward function

        Args:
            x (Tensor): (B,L,d_model)
        """
        return self.W_2(nn.functional.relu(self.W_1(x)))
