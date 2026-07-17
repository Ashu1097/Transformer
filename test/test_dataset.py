from data.dataset import GPTDataset

tokens = list(range(10))

dataset = GPTDataset(
    tokens,
    seq_len = 4
)

print("Length:", len(dataset))

for i in range(3):
    x, y = dataset[i]

    print()

    print("Input :", x.tolist())
    print("Target:", y.tolist())
