import requests
from pathlib import Path

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

text = requests.get(url).text

Path("data/tiny_shakespeare.txt").write_text(
    text,
    encoding="utf-8"
)

print("Saved!")
print(len(text))
