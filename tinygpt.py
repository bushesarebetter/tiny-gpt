import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, d, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d=128, dropout=0.1, tie=True):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(block_size, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(d, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tie:
            self.head.weight = self.tok.weight
        # small init matters with tying: nn.Embedding's N(0, 1) default gives a starting loss near 80
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1] / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, iters=20):
    model.eval()
    out = {k: sum(model(*get_batch(v, block_size, batch_size))[1].item() for _ in range(iters)) / iters
           for k, v in splits.items()}
    model.train()
    return out


def train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    text = open(a.data, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    if a.quick:
        cfg = dict(block_size=64, n_layer=3, n_head=4, d=128, dropout=0.0)
        steps, batch_size = 600, 32
    else:
        cfg = dict(block_size=96, n_layer=4, n_head=4, d=128, dropout=0.1)
        steps, batch_size = 1500, 32
    cfg["tie"] = not a.no_tie
    model = GPT(len(chars), **cfg)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, vocab {len(chars)}, "
          f"{len(text)} chars")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)

    t0 = time.time()
    for step in range(steps + 1):
        if step % (steps // 6) == 0:
            losses = estimate_loss(model, splits, cfg["block_size"], batch_size)
            print(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  "
                  f"{time.time() - t0:.0f}s")
        if step == steps:
            break
        _, loss = model(*get_batch(splits["train"], cfg["block_size"], batch_size))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    torch.save({"model": model.state_dict(), "cfg": cfg, "chars": chars}, a.ckpt)
    print(f"saved {a.ckpt}\n")
    print(sample_text(model, chars, "ROMEO:\n", 400, 0.8, 20))


def sample_text(model, chars, prompt, n, temperature, top_k):
    stoi = {c: i for i, c in enumerate(chars)}
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi] or [0]])
    out = model.generate(idx, n, temperature, top_k)
    return "".join(chars[i] for i in out[0].tolist())


def sample(a):
    torch.manual_seed(a.seed)
    ck = torch.load(a.ckpt)
    model = GPT(len(ck["chars"]), **ck["cfg"])
    model.load_state_dict(ck["model"])
    print(sample_text(model, ck["chars"], a.prompt, a.length, a.temperature, a.top_k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="character-level GPT")
    ap.add_argument("mode", choices=["train", "sample"])
    ap.add_argument("--data", default="shakespeare.txt")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--quick", action="store_true", help="smaller model, fewer steps (a few minutes on CPU)")
    ap.add_argument("--no-tie", action="store_true", help="untie input embedding and output head")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4, help="more threads ran slower on my laptop")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--length", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    a = ap.parse_args()
    train(a) if a.mode == "train" else sample(a)
