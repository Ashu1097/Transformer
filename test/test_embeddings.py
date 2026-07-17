import torch
from model.embedding import Embedding

embedding = Embedding(
    vocab_size= 10,
    embedding_dim= 4
)
token = torch.tensor([
    [1, 3, 5],
    [2, 4, 6]
])
result = embedding(token)
print(result)
print(result.shape)
