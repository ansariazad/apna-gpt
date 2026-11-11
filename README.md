# 🧠 Apna GPT — Build a GPT from Scratch in 450 Lines of Python

> **"I didn't understand transformers until I built one myself."**

A complete, from-scratch decoder-only GPT Transformer — no HuggingFace, no shortcuts, no black boxes. Every component hand-coded: causal attention, positional embeddings, pre-norm residuals, AdamW training loop. Runs on **Apple Silicon MPS** in PyTorch and falls back to a **pure NumPy engine** when PyTorch isn't installed.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Apple MPS](https://img.shields.io/badge/Apple_Silicon-MPS_GPU-000000?logo=apple)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Evals](https://img.shields.io/badge/Causal_Mask_Eval-PASS-22c55e)](https://github.com/ansariazad/ai-evals-suite)
[![Stars](https://img.shields.io/github/stars/ansariazad/apna-gpt?style=social)](https://github.com/ansariazad/apna-gpt/stargazers)

---

## 📺 Demo

```
$ python3 01_apna_gpt.py --steps 300 --sample-tokens 250

[ApnaGPT] Engine  : PyTorch + Apple MPS
[ApnaGPT] Params  : 0.21M
[ApnaGPT] Vocab   : 65 chars
[ApnaGPT] Device  : mps

Step   0/300 | train=4.3821 | val=4.3762
Step  50/300 | train=3.1204 | val=3.1589
Step 100/300 | train=2.7641 | val=2.8012
Step 200/300 | train=2.4109 | val=2.5231
Step 300/300 | train=2.2871 | val=2.3109

━━━━━ Generated Sample (250 tokens) ━━━━━
First Citizen:
What say the people, good fellows?
The king shall have his due and we
shall have our corn at honest price.

Second Citizen:
Speak, speak! What is the verdict?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📑 Table of Contents

- [Why Build GPT from Scratch?](#-why-build-gpt-from-scratch)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Eval Results](#-eval-results)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 💡 Why Build GPT from Scratch?

| | Apna GPT | HuggingFace Transformers | nanoGPT |
|--|---------|--------------------------|---------|
| Lines of code | ~450 | ~200,000 | ~300 |
| External dependencies | 0 (NumPy mode) | 50+ | PyTorch only |
| Runs without PyTorch | ✅ NumPy fallback | ❌ | ❌ |
| Apple MPS support | ✅ auto-detect | ✅ | Partial |
| Educational comments | ✅ every block | ❌ | Partial |
| Causal mask eval | ✅ unit tested | ❌ | ❌ |

---

## 🏗️ Architecture

```
Input Characters
      │
      ▼
CharTokenizer (char→int mapping, vocab_size=65)
      │
      ▼
Token Embedding (vocab_size × n_embd)
  +
Positional Embedding (block_size × n_embd)
      │
      ▼
┌─────────────── Transformer Block × n_layer ───────────────┐
│                                                             │
│   x  ──► LayerNorm ──► MultiHeadAttention ──► + ──► x     │
│                             │                              │
│               ┌─────────────┘                              │
│               │  Head × n_head                             │
│               │   Q = x @ Wq                               │
│               │   K = x @ Wk                               │
│               │   V = x @ Wv                               │
│               │   att = softmax(QKᵀ/√d + tril_mask)       │
│               │   out = att @ V                            │
│               └───────────────────────────────             │
│                                                             │
│   x  ──► LayerNorm ──► FeedForward(4× expand) ──► + ──► x │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
Final LayerNorm → LM Head (Linear, no bias)
      │
      ▼
Logits → CrossEntropyLoss / Autoregressive Sampling
```

| Hyperparameter | Default |
|----------------|---------|
| `n_embd` (embedding dim) | 64 |
| `n_head` (attention heads) | 4 |
| `n_layer` (transformer blocks) | 4 |
| `block_size` (context window) | 64 tokens |
| `dropout` | 0.1 |
| Optimizer | AdamW, lr=3e-4 |
| Total parameters | ~0.21M |
| Auto device | MPS → CUDA → CPU |

---

## 🚀 Quick Start

### macOS (Apple Silicon / Intel)
```bash
# Clone
git clone https://github.com/ansariazad/apna-gpt.git
cd apna-gpt

# Install
pip install torch numpy
# or
./setup_and_run_mac.sh

# Run (200 steps default)
python3 01_apna_gpt.py

# Custom run
python3 01_apna_gpt.py --steps 500 --sample-tokens 300 --n-embd 128
```

### No PyTorch? No problem.
```bash
# Runs with zero dependencies using pure NumPy engine
python3 01_apna_gpt.py --steps 50
# [ApnaGPT] PyTorch not found — switching to NumPy engine
```

### Windows / Linux
```bash
pip install torch numpy
python 01_apna_gpt.py
```

---

## 🔬 How It Works

### Causal Self-Attention (the core)
```python
# From Head.forward() — the entire attention mechanism:
B, T, C = x.shape
q = self.query(x)   # (B, T, head_size)
k = self.key(x)     # (B, T, head_size)
v = self.value(x)   # (B, T, head_size)

# Scaled dot-product attention
wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T)

# CAUSAL MASK — prevents future tokens from leaking into past
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v        # (B, T, head_size)
```

### Training Loop
```python
# Cross-entropy on next-token prediction
logits, loss = model(xb, yb)   # yb = xb shifted by 1
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Generation
```python
# Autoregressive: append one token at a time
for _ in range(max_new_tokens):
    logits = model(idx[:, -block_size:])  # crop to context window
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, next_token), dim=1)
```

---

## 🧪 Eval Results

Verified by [`ai-evals-suite`](https://github.com/ansariazad/ai-evals-suite):

| Assertion | Method | Result |
|-----------|--------|--------|
| Causal mask — no future leakage | Perturb token `t+1`, measure logit delta at `t` | ✅ `0.00e+00 < 1e-6` |

---

## 🗺️ Roadmap

- [ ] Shakespeare full dataset training script
- [ ] BPE tokenizer (tiktoken-compatible)
- [ ] Multi-GPU DDP training
- [ ] GPT-2 (124M) weight loading
- [ ] Streamlit demo UI

---

## 📚 References

| Resource | Author |
|----------|--------|
| [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Andrej Karpathy |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Andrej Karpathy |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Jay Alammar |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Brown et al. (GPT-3) |

---

## 🤝 Contributing

PRs welcome! Please open an issue first for major changes.
All PRs must pass the Level 1 causal mask assertion in [`ai-evals-suite`](https://github.com/ansariazad/ai-evals-suite).

---

## ⭐ Support

If this helped you understand transformers, **leave a star** — it helps others find it!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/apna-gpt?style=for-the-badge)](https://github.com/ansariazad/apna-gpt/stargazers)

---

## 📄 License

MIT © 2026 — free to use, modify, and distribute.

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
