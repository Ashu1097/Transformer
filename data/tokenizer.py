import tiktoken

class Tokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    @property
    def vocab_size(self):
        return self.encoder.n_vocab
