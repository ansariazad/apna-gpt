# 🧠 Apna GPT — Decoder-Only Transformer from Scratch

Build a GPT-style language model from scratch using pure PyTorch — no HuggingFace, no shortcuts.

## ✨ Features

- **Full Transformer decoder** — multi-head causal self-attention, positional embeddings, LayerNorm, FFN
- **Causal masking** — zero future leakage (verified by unit eval)
- **Apple Silicon MPS support** — trains on M1/M2/M3 GPU automatically
- **Character-level BPE tokenizer** — trains on any raw text corpus
- **Karpathy nanoGPT-style** architecture (clean, readable, educational)

## 🏗️ Architecture

| Component | Spec |
|-----------|------|
| Layers | 6 Transformer blocks |
| Heads | 6 attention heads |
| Embedding dim | 384 |
| Context window | 256 tokens |
| Parameters | ~10M |
| Device | MPS (Apple Silicon) → CUDA → CPU |

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch numpy

# Run the model
python 01_apna_gpt.py
```

Or use the full setup script:

```bash
chmod +x setup_and_run_mac.sh
./setup_and_run_mac.sh
```

## 📋 What It Does

1. Loads a text corpus (Shakespeare by default)
2. Trains a character-level GPT from scratch
3. Generates novel text after training
4. Verifies causal masking (no future token leakage)

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar

## 📄 License

MIT
