import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.layernorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.layer_norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x):
        attention_output, attention = self.attention(self.layer_norm1(x))
        x = x + attention_output
        x = x + self.feed_forward(self.layer_norm2(x))

        return x, attention
