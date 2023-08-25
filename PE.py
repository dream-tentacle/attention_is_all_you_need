import torch
import math


def PE(pos, embedding_dim):
    re = torch.zeros(embedding_dim)
    for i in range(embedding_dim):
        if i % 2 == 0:
            re[i] = math.sin(pos / (math.pow(10000, 2 * i / embedding_dim)))
        else:
            re[i] = math.cos(pos / (math.pow(10000, 2 * i / embedding_dim)))
    return re
