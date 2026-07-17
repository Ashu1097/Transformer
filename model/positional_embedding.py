import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        return self.embedding(positions)
