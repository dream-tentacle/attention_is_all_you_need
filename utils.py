import torch
import copy


def clone(layer, N):
    return [copy.deepcopy(layer) for _ in range(N)]
