import torch

from model.gpt import GPT
from model.config import Config

config = Config(
    vocab_size=100,
    seq_len=32,
    embedding_dim=64,
    num_heads=8,
    num_layers=4,
)

model = GPT(config)

x = torch.randint(
    0,
    config.vocab_size,
    (2, 16)
)

logits, attention = model(x)

print(logits.shape)
print(len(attention))
print(attention[0].shape)
