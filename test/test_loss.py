import torch

from model.loss import Loss

batch = 2
seq_len = 10
vocab_size = 100

logits = torch.randn(
    batch,
    seq_len,
    vocab_size
)

targets = torch.randint(
    0,
    vocab_size,
    (batch, seq_len)
)

criterion = Loss()

loss = criterion(
    logits,
    targets
)

print(loss)
