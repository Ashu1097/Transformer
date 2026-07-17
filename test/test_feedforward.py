import torch
from model.feedforward import FeedForward


embedding_dim = 64
batch = 2
seq_len = 10

feedforward = FeedForward(embedding_dim)
x = torch.randn(batch, seq_len, embedding_dim)
y = feedforward.fc1(x)
print(y.shape)
output = feedforward(x)
print(output.shape)
