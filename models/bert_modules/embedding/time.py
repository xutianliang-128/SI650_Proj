import torch.nn as nn
import torch
import math


class TimeEmbedding(nn.Embedding):
    def __init__(self, max_len, embed_size=512):
        super().__init__(max_len, embed_size, padding_idx=0)