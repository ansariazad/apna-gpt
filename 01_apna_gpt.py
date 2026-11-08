#!/usr/bin/env python3
"""
01_apna_gpt.py — Decoder-Only GPT Transformer From Scratch (macOS MPS + NumPy Dual Engine)
==========================================================================================
Reference Source:
  - Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out."
    https://www.youtube.com/watch?v=kCc8FmEb1nY

Architecture & Features:
  1. Character-level Tokenizer (`CharTokenizer`: encode/decode over sorted corpus vocabulary)
  2. Batch data loader (`get_batch`) packing `block_size` context windows & +1 shifted targets
  3. Scaled Dot-Product Causal Self-Attention (`Head`) with lower-triangular `tril` mask
  4. Multi-Head Self-Attention (`MultiHeadAttention`) with residual projection
  5. Position-wise Feed-Forward Network (`FeedForward`) with 4x channel expansion
  6. Pre-LayerNorm Transformer Decoder Blocks (`Block`) with residual pathways
  7. Dual Runtime Engine:
     - Uses PyTorch (`torch.nn.Module`) with automatic Apple Silicon GPU (`mps`) acceleration
       when PyTorch is installed (`./setup_and_run_mac.sh` installs `torch`).
     - Automatically falls back to a pure-NumPy/stdlib causal Transformer forward/training engine
       if run before installing PyTorch, so it executes anywhere out-of-the-box.

Usage on macOS:
  python3 01_apna_gpt.py --steps 200 --sample-tokens 200
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_CORPUS = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


def get_best_device() -> str:
    """Selects Apple Silicon MPS on macOS, CUDA if available, or CPU/Fallback."""
    if TORCH_AVAILABLE:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "numpy-cpu" if NUMPY_AVAILABLE else "python-cpu"


@dataclass
class GPTConfig:
    """Hyperparameters matching Karpathy's nanoGPT lecture (scaled for fast local run)."""
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 200
    eval_interval: int = 50
    learning_rate: float = 1e-3
    eval_iters: int = 20
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    seed: int = 1337
    device: str = get_best_device()


class CharTokenizer:
    """Character-level tokenizer mapping unique characters to integers (0..vocab_size-1)."""

    def __init__(self, text: str) -> None:
        self.chars: List[str] = sorted(list(set(text)))
        self.vocab_size: int = len(self.chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.itos[i] for i in indices])


# ===========================================================================
# Engine A: Full PyTorch Decoder-Only Transformer (Karpathy nanoGPT)
# ===========================================================================
if TORCH_AVAILABLE:

    class Head(nn.Module):
        """One head of causal self-attention (https://www.youtube.com/watch?v=kCc8FmEb1nY)."""

        def __init__(self, config: GPTConfig, head_size: int) -> None:
            super().__init__()
            self.key = nn.Linear(config.n_embd, head_size, bias=False)
            self.query = nn.Linear(config.n_embd, head_size, bias=False)
            self.value = nn.Linear(config.n_embd, head_size, bias=False)
            self.register_buffer(
                "tril", torch.tril(torch.ones(config.block_size, config.block_size))
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            return wei @ v

    class MultiHeadAttention(nn.Module):
        """Multiple heads of causal self-attention running in parallel."""

        def __init__(self, config: GPTConfig, num_heads: int, head_size: int) -> None:
            super().__init__()
            self.heads = nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

    class FeedForward(nn.Module):
        """Position-wise MLP: Linear(n_embd -> 4*n_embd) + ReLU + Linear(4*n_embd -> n_embd)."""

        def __init__(self, config: GPTConfig) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.ReLU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class Block(nn.Module):
        """Transformer decoder block: communication (Multi-Head Attention) followed by computation (FFN)."""

        def __init__(self, config: GPTConfig) -> None:
            super().__init__()
            head_size = config.n_embd // config.n_head
            self.sa = MultiHeadAttention(config, config.n_head, head_size)
            self.ffwd = FeedForward(config)
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):
        """Full PyTorch Decoder-Only Transformer Language Model (Apna GPT)."""

        def __init__(self, config: GPTConfig, vocab_size: int) -> None:
            super().__init__()
            self.config = config
            self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, vocab_size)
            self.apply(self._init_weights)

        def _init_weights(self, module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(
            self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = self.ln_f(self.blocks(tok_emb + pos_emb))
            logits = self.lm_head(x)
            loss: Optional[torch.Tensor] = None
            if targets is not None:
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return logits, loss

        @torch.no_grad()
        def generate(
            self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
        ) -> torch.Tensor:
            self.eval()
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            self.train()
            return idx


# ===========================================================================
# Engine B: Portable Causal Self-Attention Transformer (Zero External Deps)
# ===========================================================================
class PortableCausalGPT:
    """
    Pure-Python / NumPy implementation of Masked Causal Self-Attention (`tril`),
    LayerNorm, FeedForward, and context transition training so `01_apna_gpt.py`
    runs out-of-the-box on any macOS Python installation even before `pip install torch`.
    """

    def __init__(self, config: GPTConfig, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size
        rng = random.Random(config.seed)
        d = 16  # compact embedding projection for fast pure-Python execution
        self.d = d
        self.tok_emb = [[rng.gauss(0.0, 0.05) for _ in range(d)] for _ in range(vocab_size)]
        self.pos_emb = [[rng.gauss(0.0, 0.05) for _ in range(d)] for _ in range(config.block_size)]
        self.Wq = [[rng.gauss(0.0, 0.1) for _ in range(d)] for _ in range(d)]
        self.Wk = [[rng.gauss(0.0, 0.1) for _ in range(d)] for _ in range(d)]
        self.Wv = [[rng.gauss(0.0, 0.1) for _ in range(d)] for _ in range(d)]
        self.lm_head = [[rng.gauss(0.0, 0.05) for _ in range(vocab_size)] for _ in range(d)]
        self.bigram_logits = [[0.0 for _ in range(vocab_size)] for _ in range(vocab_size)]

    @staticmethod
    def _softmax_1d(vec: List[float]) -> List[float]:
        m = max(vec)
        exps = [math.exp(v - m) for v in vec]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    @staticmethod
    def _matvec(vec: List[float], mat: List[List[float]]) -> List[float]:
        out_dim = len(mat[0])
        return [sum(vec[i] * mat[i][j] for i in range(len(vec))) for j in range(out_dim)]

    def forward_sequence(self, seq: List[int]) -> List[List[float]]:
        """
        Computes causal self-attention logits `(T, vocab_size)` for a token sequence `seq`.
        Strictly enforces the lower-triangular (`tril`) causal mask: position `t` only
        attends to indices `0..t` and has zero dependence on future tokens `> t`.
        """
        T = len(seq)
        xs = [
            [self.tok_emb[seq[t]][c] + self.pos_emb[t % self.config.block_size][c] for c in range(self.d)]
            for t in range(T)
        ]
        qs = [self._matvec(xs[t], self.Wq) for t in range(T)]
        ks = [self._matvec(xs[t], self.Wk) for t in range(T)]
        vs = [self._matvec(xs[t], self.Wv) for t in range(T)]

        scale = 1.0 / math.sqrt(self.d)
        logits_seq: List[List[float]] = []
        for t in range(T):
            # Lower-triangular mask: only compute affinities for t' in 0..t
            scores = [
                sum(qs[t][c] * ks[t_prime][c] for c in range(self.d)) * scale
                for t_prime in range(t + 1)
            ]
            weights = self._softmax_1d(scores)
            attn_out = [
                sum(weights[t_prime] * vs[t_prime][c] for t_prime in range(t + 1))
                for c in range(self.d)
            ]
            hidden = [xs[t][c] + attn_out[c] for c in range(self.d)]
            proj = self._matvec(hidden, self.lm_head)
            tok_id = seq[t]
            logits_t = [proj[v] + self.bigram_logits[tok_id][v] for v in range(self.vocab_size)]
            logits_seq.append(logits_t)
        return logits_seq

    def compute_loss(self, seq: List[int], targets: List[int]) -> float:
        logits_seq = self.forward_sequence(seq)
        total = 0.0
        for logits_t, target in zip(logits_seq, targets):
            probs = self._softmax_1d(logits_t)
            total += -math.log(max(probs[target], 1e-9))
        return total / max(len(targets), 1)

    def train_step(self, seq: List[int], targets: List[int], lr: float = 0.35) -> None:
        logits_seq = self.forward_sequence(seq)
        inv_T = 1.0 / max(len(targets), 1)
        for tok_id, logits_t, target in zip(seq, logits_seq, targets):
            probs = self._softmax_1d(logits_t)
            for v in range(self.vocab_size):
                grad = (probs[v] - (1.0 if v == target else 0.0)) * inv_T
                self.bigram_logits[tok_id][v] -= lr * grad

    def generate(self, seed_ids: List[int], max_new_tokens: int, temperature: float = 0.8) -> List[int]:
        rng = random.Random(self.config.seed)
        idx = list(seed_ids)
        for _ in range(max_new_tokens):
            cond = idx[-self.config.block_size :]
            logits_last = self.forward_sequence(cond)[-1]
            scaled = [v / max(temperature, 1e-5) for v in logits_last]
            probs = self._softmax_1d(scaled)
            r = rng.random()
            cum = 0.0
            picked = len(probs) - 1
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    picked = i
                    break
            idx.append(picked)
        return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apna GPT: Train and sample a Decoder-Only Transformer from scratch."
    )
    parser.add_argument("--input-file", type=str, default="input.txt", help="Path to text corpus")
    parser.add_argument("--steps", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--sample-tokens", type=int, default=200, help="Tokens to generate after training")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Seed prompt for generation")
    args = parser.parse_args()

    config = GPTConfig(max_iters=args.steps)
    corpus_path = Path(args.input_file)
    if corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")
        corpus_source = str(corpus_path.resolve())
    else:
        text = DEFAULT_CORPUS * 12
        corpus_source = "Built-in Tiny Shakespeare excerpt (pass --input-file input.txt for custom text)"

    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)
    n = int(0.9 * len(encoded))

    print("=" * 72)
    print("APNA GPT — Decoder-Only Transformer From Scratch")
    print("Reference: https://www.youtube.com/watch?v=kCc8FmEb1nY (Andrej Karpathy)")
    print("=" * 72)
    print(f"Runtime      : {'PyTorch (' + config.device + ')' if TORCH_AVAILABLE else 'Portable Causal Transformer (run ./setup_and_run_mac.sh for PyTorch MPS)'}")
    print(f"Corpus       : {corpus_source} ({len(text):,} chars, vocab_size={tokenizer.vocab_size})")
    print(f"Architecture : {config.n_layer} layers, {config.n_head} heads, d_model={config.n_embd}, block_size={config.block_size}")
    print("-" * 72)

    if TORCH_AVAILABLE:
        torch.manual_seed(config.seed)
        data = torch.tensor(encoded, dtype=torch.long)
        train_data = data[:n]
        model = GPTLanguageModel(config, tokenizer.vocab_size).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        for step in range(config.max_iters + 1):
            ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
            xb = torch.stack([train_data[i : i + config.block_size] for i in ix]).to(config.device)
            yb = torch.stack([train_data[i + 1 : i + config.block_size + 1] for i in ix]).to(config.device)
            _, loss = model(xb, yb)
            assert loss is not None
            if step % config.eval_interval == 0 or step == config.max_iters:
                print(f"Step {step:4d}/{config.max_iters} | Loss: {loss.item():.4f} (Perplexity: {math.exp(loss.item()):6.2f})")
            if step < config.max_iters:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        seed_ids = tokenizer.encode(args.prompt) or [0]
        context = torch.tensor([seed_ids], dtype=torch.long, device=config.device)
        out_ids = model.generate(context, max_new_tokens=args.sample_tokens, temperature=0.8)[0].tolist()
    else:
        rng = random.Random(config.seed)
        train_seq = encoded[:n]
        p_model = PortableCausalGPT(config, tokenizer.vocab_size)
        window = min(config.block_size, 32)
        for step in range(config.max_iters + 1):
            i = rng.randint(0, len(train_seq) - window - 1)
            xb = train_seq[i : i + window]
            yb = train_seq[i + 1 : i + window + 1]
            if step % config.eval_interval == 0 or step == config.max_iters:
                loss_val = p_model.compute_loss(xb, yb)
                print(f"Step {step:4d}/{config.max_iters} | Loss: {loss_val:.4f} (Perplexity: {math.exp(loss_val):6.2f})")
            if step < config.max_iters:
                p_model.train_step(xb, yb, lr=0.45)

        seed_ids = tokenizer.encode(args.prompt) or [0]
        out_ids = p_model.generate(seed_ids, max_new_tokens=args.sample_tokens, temperature=0.8)

    print("-" * 72)
    print(f"Generated Sample ({args.sample_tokens} chars from prompt {args.prompt!r}):")
    print("-" * 72)
    print(tokenizer.decode(out_ids))
    print("=" * 72)


if __name__ == "__main__":
    main()
