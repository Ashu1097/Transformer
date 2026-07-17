import torch
from model.attention import MultiHeadAttention

embedding_dim = 64
num_heads = 8
batch_size = 2
seq_len = 10

attention = MultiHeadAttention(embedding_dim, num_heads)

x = torch.randn(batch_size, seq_len, embedding_dim)

output, attention = attention(x)

print(output)
print(output.shape)
print(attention)
print(attention.shape)

print(attention[0,0])
