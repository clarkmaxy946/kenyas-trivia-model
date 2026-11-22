"""A small GPT (decoder-only transformer), written from scratch in PyTorch."""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1


PRESETS = {  # ~parameter counts with the 8k vocabulary
    "tiny": dict(n_layer=2, n_head=2, n_embd=64),      # smoke tests only
    "small": dict(n_layer=8, n_head=8, n_embd=512),     # ~29M  (first scaling stage)
    "base": dict(n_layer=14, n_head=12, n_embd=768),    # ~105M (second scaling stage)
}


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.drop = c.dropout

    def forward(self, x, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if past is None:  # training / first pass over the prompt
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        else:  # generation: one new token attends to all cached keys/values
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C)), (k, v)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd, bias=False))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, past=None):
        a, kv = self.attn(self.ln1(x), past)
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.ln2(x))), kv


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tie input/output embeddings
        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled init of residual projections
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past=None):
        B, T = idx.shape
        off = 0 if past is None else past[0][0].size(2)
        x = self.drop(self.tok(idx) + self.pos(torch.arange(off, off + T, device=idx.device)))
        kvs = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, None if past is None else past[i])
            kvs.append(kv)
        logits = self.head(self.ln_f(x))
        self.last_kv = kvs
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, eos_id, temperature=0.8, top_k=40):
        """Sample tokens one at a time, reusing cached keys/values (fast on CPU)."""
        logits, _ = self(idx[:, -self.c.block_size:])
        past = self.last_kv
        for _ in range(max_new):
            if idx.size(1) >= self.c.block_size:
                break
            l = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(l, top_k)
                l[l < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(l, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt[0, 0]) == eos_id:
                break
            logits, _ = self(nxt, past=past)
            past = self.last_kv
        return idx


def build(preset, vocab_size, block_size, dropout=0.1):
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size, dropout=dropout, **PRESETS[preset]))


def config_dict(model):
    return asdict(model.c)
