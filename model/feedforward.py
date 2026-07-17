import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x
