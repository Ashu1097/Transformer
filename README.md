# 🚀 Transformer PyTorch

A  GPT language model implemented completely from  using **PyTorch**.

This project recreates the core architecture behind GPT-style language models without relying on Hugging Face's model implementations. Every major component of the Transformer decoder is implemented manually for educational purposes.

---

## ✨ Features

- ✅ GPT Decoder Architecture
- ✅ Token Embeddings
- ✅ Learned Positional Embeddings
- ✅ Multi-Head Self Attention
- ✅ Feed Forward Network
- ✅ Layer Normalization
- ✅ Residual Connections
- ✅ Weight Tying
- ✅ GPT-2 Tokenizer (`tiktoken`)
- ✅ Tiny Shakespeare Dataset
- ✅ Training Pipeline
- ✅ Checkpoint Saving & Resume Training
- ✅ Text Generation
- ✅ Temperature Sampling
- ✅ Top-k Sampling
- ✅ Top-p (Nucleus) Sampling
- ✅ Unit Tests

---

# 🏗️ Architecture

```
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Positional Embedding
      │
      ▼
Transformer Block × N
      │
      ├── Multi-Head Self Attention
      ├── Feed Forward Network
      ├── Residual Connections
      └── LayerNorm
      │
      ▼
Final LayerNorm
      │
      ▼
Linear Head (Weight Tied)
      │
      ▼
Vocabulary Logits
```

---

# 📁 Project Structure

```
Transformer/
│
├── data/
│   ├── dataset.py
│   ├── tokenizer.py
│   └── tiny_shakespeare.txt
│
├── model/
│   ├── attention.py
│   ├── config.py
│   ├── embedding.py
│   ├── feedforward.py
│   ├── gpt.py
│   ├── layernorm.py
│   ├── loss.py
│   ├── positional_embedding.py
│   └── transformer_block.py
│
├── test/
│
├── train.py
├── trainer.py
├── generate.py
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# ⚙️ Model Configuration

| Parameter | Value |
|-----------|------:|
| Vocabulary Size | 50257 |
| Embedding Size | 128 |
| Layers | 4 |
| Attention Heads | 4 |
| Context Length | 64 |
| Optimizer | AdamW |

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Transformer.git
cd Transformer
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python train.py
```

Training automatically:

- Creates checkpoints
- Saves model state
- Can resume from checkpoints

---

# ✍️ Generate Text

```bash
python generate.py
```

Example:

```
Prompt:

To be

Output:

To be so short but the earth,
And yet the earth had not precise in't,
Nor can these sorrows...
```

---

# 🧪 Running Tests

```bash
pytest
```

---

# 🧠 Decoding Methods

The project supports multiple decoding strategies:

- Greedy Decoding
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus Sampling)

Example:

```python
output = model.generate(
    input_ids,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
```

---

# 🚀 Future Improvements

- [ ] Validation Loop
- [ ] Learning Rate Scheduler
- [ ] TensorBoard Logging
- [ ] Mixed Precision (AMP)
- [ ] Attention Visualization
- [ ] KV Cache
- [ ] RMSNorm
- [ ] Rotary Positional Embeddings (RoPE)
- [ ] SwiGLU FeedForward
- [ ] Flash Attention
- [ ] Hugging Face Compatibility
- [ ] Train on TinyStories
- [ ] Train on OpenWebText

---

# 📚 What I Learned

This project helped me understand:

- Transformer Architecture
- Self-Attention
- Multi-Head Attention
- Position Embeddings
- Autoregressive Language Modeling
- Tokenization
- Training Large Language Models
- Sampling Strategies
- PyTorch Module Design
- Checkpointing
- Unit Testing

---

# 📄 License

MIT License

---

## Acknowledgements

Special thanks to **ChatGPT (OpenAI)** and Documentations for helping explain Transformer concepts, reviewing implementations, and assisting with debugging throughout the development of this project.

# ⭐ If you found this project useful, consider giving it a star!
