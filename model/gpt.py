import torch
import torch.nn as nn
from model.embedding import Embedding
from model.positional_embedding import PositionalEmbedding
from model.layernorm import LayerNorm
from model.transformer_block import TransformerBlock
from model.config import Config

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = PositionalEmbedding(config.seq_len, config.embedding_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.embedding_dim, config.num_heads) for _ in range(config.num_layers)])
        self.ln_final = LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids):
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(input_ids)
        x = token_embeddings + position_embeddings
        attention_map = []
        for block in self.transformer_blocks:
            x, attention = block(x)
            attention_map.append(attention)
        # Final LayerNorm
        x = self.ln_final(x)
        # Vocabulary logits
        logits = self.lm_head(x)
        return logits, attention_map

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens,temperature=1.0,top_k=None,top_p=None,):
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError("top_p must be between 0 and 1.")
        for _ in range(max_new_tokens):
            input_context = input_ids[:, -self.config.seq_len:]
            logits, _ = self(input_context)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(
                    logits < values[:, [-1]],
                    float("-inf"),
                )
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                )

                sorted_probs = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1,
                )

                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(
                    logits,
                    dtype=torch.bool,
                )
                indices_to_remove.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )
                logits = logits.masked_fill(
                    indices_to_remove,
                    float("-inf"),
                )
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat(
                [input_ids, next_token],
                dim =1
            )

        return input_ids
