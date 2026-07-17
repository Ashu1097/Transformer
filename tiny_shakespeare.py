from pathlib import Path

from data.tokenizer import Tokenizer

tokenizer = Tokenizer()

text = Path(
    "data/tiny_shakespeare.txt"
).read_text(
    encoding="utf-8"
)

tokens = tokenizer.encode(text)

print(len(tokens))
