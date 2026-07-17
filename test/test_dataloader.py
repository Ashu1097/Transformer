from torch.utils.data import DataLoader
from data.dataset import GPTDataset

tokens = list(range(20))

dataset = GPTDataset(
    tokens=tokens,
    seq_len=5
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for inputs, targets in train_loader:
    print("Inputs:")
    print(inputs)

    print("\nTargets:")
    print(targets)

    break
