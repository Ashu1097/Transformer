import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps

        # Learnable scale parameter (γ)
        self.gamma = nn.Parameter(torch.ones(embedding_dim))

        # Learnable shift parameter (β)
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.gamma * x + self.beta
        return x
