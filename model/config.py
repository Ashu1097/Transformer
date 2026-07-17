from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    seq_len: int
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    bias: bool = False
