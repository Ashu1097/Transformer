import torch

from model.config import Config
from model.gpt import GPT
from data.tokenizer import Tokenizer

config = Config(
    vocab_size=50257,
    seq_len=64,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
)

model = GPT(config)

checkpoint = torch.load(
    "checkpoints/gpt_7.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

tokenizer = Tokenizer()

prompt = "To be"

tokens = tokenizer.encode(prompt)

input_ids = torch.tensor(
    [tokens],
    dtype=torch.long,
)

output = model.generate(
    input_ids=input_ids,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)

generated = tokenizer.decode(
    output[0].tolist()
)

print(generated)
