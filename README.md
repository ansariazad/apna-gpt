# Apna GPT

**ApnaGPT** is a fully hand-coded, decoder-only GPT Transformer trained from scratch in pure PyTorch — no HuggingFace, no shortcuts. Implements every component of the original *Attention Is All You Need* decoder stack with Apple Silicon MPS acceleration, plus a zero-dependency NumPy fallback engine so it runs anywhere out-of-the-box.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-black?logo=apple&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Evals-100%25_Pass-brightgreen)

<img width="800" alt="ApnaGPT Training" src="https://raw.githubusercontent.com/karpathy/nanoGPT/master/assets/nanogpt.jpg" />

---

## What is this?

ApnaGPT is an educational implementation of the GPT architecture directly following Andrej Karpathy's ["Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture. Every component is coded from first principles:

- The tokenizer is a simple character-level encoder (sorted vocabulary, no BPE library)
- The attention mask is a hand-built `torch.tril` lower-triangular matrix — no `F.scaled_dot_product_attention` wrappers
- The training loop uses raw `AdamW` with manually computed cross-entropy loss
- A complete NumPy/stdlib fallback engine runs the same forward pass without PyTorch installed at all

This is built for **deep understanding**, not production throughput.

---

## Key Features

- **Causal Self-Attention from scratch** — `Head` class implements scaled dot-product attention with explicit `tril` mask; zero future-token leakage (verified by unit eval)
- **Multi-Head Attention** — `MultiHeadAttention` aggregates 6 parallel heads with a linear projection
- **Pre-LayerNorm Residual Blocks** — `Block` class: `x = x + attn(ln1(x))` then `x = x + ffn(ln2(x))`
- **Dual Runtime Engine** — automatically uses PyTorch + Apple MPS when available, falls back to pure NumPy/stdlib so the script runs anywhere with zero install
- **Apple Silicon MPS acceleration** — trains on M1/M2/M3 GPU automatically via `torch.device("mps")`
- **CLI flags** — `--steps`, `--sample-tokens`, `--block-size`, `--n-embd` for quick experiments
- **Character-level tokenizer** — `CharTokenizer.encode()` / `.decode()` over the corpus vocabulary

---

## Architecture

```
Input Tokens (char-level)
        │
        ▼
  Token Embedding (C)  +  Positional Embedding (T)
        │
        ▼
  ┌─────────────────────────────────────┐
  │   Transformer Block × n_layer       │
  │                                     │
  │   ┌─────────────────────────────┐   │
  │   │  LayerNorm (pre-norm)       │   │
  │   │  MultiHeadAttention (n_head)│   │
  │   │    └─ Head × n_head         │   │
  │   │         Causal tril mask    │   │
  │   │  Residual Add               │   │
  │   └─────────────────────────────┘   │
  │   ┌─────────────────────────────┐   │
  │   │  LayerNorm (pre-norm)       │   │
  │   │  FeedForward (4× expansion) │   │
  │   │  Residual Add               │   │
  │   └─────────────────────────────┘   │
  └─────────────────────────────────────┘
        │
        ▼
  Final LayerNorm  →  LM Head (Linear, no bias)
        │
        ▼
  Logits (B, T, vocab_size)  →  Cross-Entropy Loss
```

| Hyperparameter | Default Value |
|----------------|---------------|
| `n_embd` | 64 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `block_size` | 64 (context window) |
| `dropout` | 0.1 |
| Optimizer | AdamW, lr=3e-4 |
| Device | MPS → CUDA → CPU |

---

## Prerequisites

**Python 3.11+** is required.

### macOS (Apple Silicon / Intel)
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
winget install Python.Python.3.11
pip install torch numpy
```

### Linux
```bash
sudo apt install python3.11 python3-pip   # Debian/Ubuntu
pip install torch numpy
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ansariazad/apna-gpt.git
cd apna-gpt

# 2. Install dependencies (macOS, with uv)
chmod +x setup_and_run_mac.sh
./setup_and_run_mac.sh

# OR manually:
pip install torch numpy

# 3. Run with defaults (200 training steps)
python3 01_apna_gpt.py

# 4. Custom run
python3 01_apna_gpt.py --steps 500 --sample-tokens 300 --n-embd 128
```

---

## Usage

### Basic Training + Generation
```bash
python3 01_apna_gpt.py --steps 200 --sample-tokens 200
```

**Expected output:**
```
[ApnaGPT] Device: mps  |  vocab_size: 65  |  params: ~0.21M
Step    0 | train loss 4.3821 | val loss 4.3762
Step  100 | train loss 2.8541 | val loss 2.9014
Step  200 | train loss 2.4109 | val loss 2.5231

--- Generated Sample (200 tokens) ---
First Citizen:
Wher hav I ben, the kinge of sorrow,
To the great lord and his men...
```

### NumPy Fallback Mode (no PyTorch needed)
```bash
# Works even without torch installed
python3 01_apna_gpt.py --steps 50
# [ApnaGPT] PyTorch not found — using pure NumPy engine
```

### CLI Options
```
--steps          Number of training iterations  (default: 200)
--sample-tokens  Tokens to generate after train  (default: 200)
--block-size     Context window length           (default: 64)
--n-embd         Embedding dimension             (default: 64)
```

---

## How It Works

### 1. Tokenizer
`CharTokenizer` builds a vocabulary from all unique characters in the corpus, encodes strings to integer lists, decodes back — no external tokenizer library needed.

### 2. Causal Self-Attention
```python
# In Head.forward():
wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T)
wei = wei.masked_fill(tril == 0, float('-inf')) # causal mask
wei = F.softmax(wei, dim=-1)
out = wei @ v
```
The `tril` mask ensures token at position `t` only sees tokens `0..t` — no future leakage.

### 3. Training Loop
AdamW optimizer, cross-entropy loss on next-token prediction. Runs `--steps` iterations with periodic validation loss reporting.

### 4. Generation
Autoregressive sampling — appends one token at a time by sampling from the softmax distribution over the vocabulary.

---

## File Structure

```
apna-gpt/
├── 01_apna_gpt.py      # Full GPT implementation (tokenizer, model, trainer, generator)
├── setup_and_run_mac.sh # One-shot macOS setup: uv venv + pip install + run
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Ignores .venv, __pycache__, *.pt, *.bin
└── README.md           # This file
```

---

## Eval Results

This module is tested by the companion [`ai-evals-suite`](https://github.com/ansariazad/ai-evals-suite):

| Assertion | Result | Detail |
|-----------|--------|--------|
| `L1_GPT_Causal_Mask_No_Future_Leakage` | ✅ PASS | Leakage = `0.00e+00 < 1e-6` |

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. (GPT-2)

---

## Contributing

Pull requests welcome! For major changes, open an issue first. All PRs must pass the Level 1 eval assertions in `ai-evals-suite`.

---

## License

MIT © 2026

---

## Author

**Azad Ansari** — [GitHub](https://github.com/ansariazad) | [AutoShorts](https://github.com/ansariazad/AutoShorts)
