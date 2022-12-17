import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class Attention2(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    # def forward(self, query, key, value, mask=None, dropout=None):
    #
    #     scores = torch.matmul(query, key.transpose(-2, -1)) \
    #              / math.sqrt(query.size(-1))
    #
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #
    #     p_attn = F.softmax(scores, dim=-1)
    #
    #     if dropout is not None:
    #         p_attn = dropout(p_attn)
    #
    #     return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, r, r_bias, r_w_bias, mask=None, dropout=None):

        rw_head_q = query + r_w_bias  # r_w_bias [n_head, d_head]
        rr_head_q = query + r_bias

        rw_head_q = rw_head_q.transpose(1, 2)
        rr_head_q = rr_head_q.transpose(1, 2)
        key = key.transpose(1,2)
        r = r.view(r.shape[0], -1, rr_head_q.shape[1], rr_head_q.shape[3])
        r = r.transpose(1, 2)

#可能有点问题, dim 上面
        # AC = torch.einsum('ibnd,jbnd->ijbn', rw_head_q, key)
        # BD = torch.einsum('ibnd,jnd->ijbn', rr_head_q, r)

        AC = torch.matmul(rw_head_q, key.transpose(-2, -1))
        BD = torch.matmul(rr_head_q, r.transpose(-2, -1))

        scores = (AC + BD)/ math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value.transpose(1,2)), p_attn
