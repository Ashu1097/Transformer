from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import GPTDataset
from data.tokenizer import Tokenizer
from model.config import Config
from model.gpt import GPT
from trainer import Trainer

text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8")
tokenizer = Tokenizer()
tokens = tokenizer.encode(text)

print(f"Tokens: {len(tokens)}")

config = Config(
    vocab_size=tokenizer.vocab_size,
    seq_len=64,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
)

dataset = GPTDataset(
    tokens=tokens,
    seq_len=config.seq_len,
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

model = GPT(config)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    config=config,
)


trainer.resume_training(
    epochs=7,
)
