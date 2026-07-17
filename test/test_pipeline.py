from pathlib import Path

from torch.utils.data import DataLoader

from data.tokenizer import Tokenizer
from data.dataset import GPTDataset

text = Path("data/tiny_shakespeare.txt").read_text(
    encoding="utf-8"
)

tokenizer = Tokenizer()

tokens = tokenizer.encode(text)

print("Number of tokens:", len(tokens))

dataset = GPTDataset(
    tokens=tokens,
    seq_len=128
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

inputs, targets = next(iter(loader))

print(inputs.shape)
print(targets.shape)
