from data.tokenizer import Tokenizer

tokenizer = Tokenizer()

text = "To be or not to be."

tokens = tokenizer.encode(text)

print("Text:")
print(text)

print()

print("Tokens:")
print(tokens)
print()

decoded = tokenizer.decode(tokens)

print("Decoded:")
print(decoded)

print()

print("Vocabulary Size:")
print(tokenizer.vocab_size)

print(tokenizer.encode("Hello"))
