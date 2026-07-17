
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024))
        )

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        # (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        assert Q.shape == (batch, seq_len, self.embedding_dim), "Q shape mismatch"
        assert K.shape == (batch, seq_len, self.embedding_dim), "K shape mismatch"
        assert V.shape == (batch, seq_len, self.embedding_dim), "V shape mismatch"

        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        assert Q.shape == (batch, self.num_heads, seq_len, self.head_dim)

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )
        assert scores.shape == (batch, self.num_heads, seq_len, seq_len)
        scores = scores * (self.head_dim ** -0.5)
        mask = getattr(self, "mask")
        mask = self.mask[:seq_len, :seq_len]
        scores.masked_fill_(mask == 0, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        output = output.transpose(1,2)
        output = output.contiguous()
        output = output.reshape(
            batch, seq_len, self.embedding_dim
        )
        output = self.out(output)
        return output, attention
