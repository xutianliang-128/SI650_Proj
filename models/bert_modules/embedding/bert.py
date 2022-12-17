import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .rating import RatingEmbedding
from .time import TimeEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, rating_num, time_num, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.rating = RatingEmbedding(max_len=rating_num, embed_size=embed_size)
        self.time = TimeEmbedding(max_len=time_num, embed_size=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, r, t):
        #pos = self.position(sequence)
        tok = self.token(sequence)
        rat = self.rating(r)
        tim = self.time(t)
        x = tok + tim + rat # + tim + rat
        return self.dropout(x), tim
