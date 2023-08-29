import torch.nn as nn
import copy


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
