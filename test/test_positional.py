import torch
from model.positional_embedding import PositionalEmbedding

batch = 2
seq = 5
embed = 4

token_embeddings = torch.randn(batch, seq, embed)

position = PositionalEmbedding(20, embed)

position_embeddings = position(seq)

x = token_embeddings + position_embeddings

print(x.shape)

#position = PositionalEmbedding(
#    max_seq_length=10,
#    embedding_dim=4
#)
#
#pos = position(5)
#
#print(pos)
#print(pos.shape)
