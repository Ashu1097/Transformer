import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index):
        input_ids = self.tokens[index:index + self.seq_len]
        target = self.tokens[index + 1:index + self.seq_len + 1]
        return (
            torch.tensor(input_ids),
            torch.tensor(target)
        )
