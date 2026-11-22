#!/usr/bin/env python3
"""
Train the card model. Two stages, run in this order:

  1. --stage pretrain   learn English + Kenyan context from the Wikipedia articles
  2. --stage cards      learn the card format from fresh random cards built from verified entries
                        (start from the pretrain weights with --init-from)

Safe on free Kaggle/Colab: it saves a checkpoint every --save-every steps and
automatically RESUMES from it if the session is restarted (just run the same command again).

Example (Kaggle, GPU on):
  python train/train.py --stage pretrain --preset small --data /kaggle/input/kt-corpus --out /kaggle/working/run1 --max-steps 3000
  python train/train.py --stage cards    --preset small --data /kaggle/input/kt-corpus --out /kaggle/working/run1 --max-steps 1500 --init-from /kaggle/working/run1/best_pretrain.pt
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cards  # noqa: E402
import model as M  # noqa: E402


def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    return tok, tok.token_to_id(cards.PAD), tok.token_to_id(cards.BOS), tok.token_to_id(cards.EOS)


def pretrain_bins(data, tok, cache):
    """Tokenise the article text once and cache it as uint16 arrays in the output folder
    (the input folder is read-only on Kaggle)."""
    out = {}
    for name in ("train", "val"):
        txt, binf = Path(data) / f"pretrain_{name}.txt", Path(cache) / f"pretrain_{name}.bin"
        if not binf.exists():
            ids = tok.encode(open(txt, encoding="utf-8").read()).ids
            np.array(ids, dtype=np.uint16).tofile(binf)
        out[name] = np.memmap(binf, dtype=np.uint16, mode="r")
    return out


def get_pretrain_batch(arr, bs, T, device):
    ix = np.random.randint(0, len(arr) - T - 1, bs)
    x = torch.from_numpy(np.stack([arr[i:i + T] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
    return x.to(device), y.to(device)


def encode_cards(texts, tok, bos, eos, pad, T, device):
    seqs = [[bos] + e.ids[:T - 2] + [eos] for e in tok.encode_batch(texts)]
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    y = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    return x[:, :-1].contiguous().to(device), y[:, :-1].contiguous().to(device)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = (step - a.warmup) / max(1, a.max_steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "cards"], required=True)
    ap.add_argument("--preset", default="small", choices=list(M.PRESETS))
    ap.add_argument("--data", required=True, help="folder with the data/corpus files")
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default=None, help="weights from the pretrain stage")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6, help="stop after this many evals without improvement")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, pad, bos, eos = load_tokenizer(a.tokenizer)
    net = M.build(a.preset, tok.get_vocab_size(), a.block_size, a.dropout).to(device)
    print(f"{a.stage} | preset {a.preset} | {net.n_params() / 1e6:.1f}M parameters | device {device}", flush=True)

    decay = [p for p in net.parameters() if p.dim() >= 2]
    nodecay = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = out / f"ckpt_{a.stage}.pt"
    best_path = out / f"best_{a.stage}.pt"
    step, best, bad = 0, float("inf"), 0
    if ckpt_path.exists():  # resume after a session restart
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, bad = ck["step"], ck["best"], ck["bad"]
        print(f"RESUMED from {ckpt_path} at step {step} (best val {best:.4f})", flush=True)
    elif a.init_from:
        net.load_state_dict(torch.load(a.init_from, map_location=device)["model"])
        print(f"initialised weights from {a.init_from}", flush=True)

    if a.stage == "pretrain":
        bins = pretrain_bins(a.data, tok, out)
        print(f"pretrain tokens: train {len(bins['train']) / 1e6:.2f}M, val {len(bins['val']) / 1e6:.2f}M", flush=True)
    else:
        entries = [json.loads(l) for l in open(Path(a.data) / "entries_train.jsonl", encoding="utf-8") if l.strip()]
        sample = cards.make_sampler(entries)
        rng = cards.rng_from(a.seed + step)
        val_texts = [l.strip() for l in open(Path(a.data) / "val_cards.txt", encoding="utf-8") if l.strip()]
        print(f"card stage: {len(entries)} training entries, {len(val_texts)} fixed val cards", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval()
        losses = []
        if a.stage == "pretrain":
            for _ in range(30):
                x, y = get_pretrain_batch(bins["val"], a.batch_size, a.block_size, device)
                with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                    losses.append(net(x, y)[1].item())
        else:
            for i in range(0, len(val_texts), a.batch_size):
                x, y = encode_cards(val_texts[i:i + a.batch_size], tok, bos, eos, pad, a.block_size, device)
                with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                    losses.append(net(x, y)[1].item())
        net.train()
        return sum(losses) / len(losses)

    def save(path, final=False):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict() if not final else None, "step": step,
                    "best": best, "bad": bad, "config": M.config_dict(net), "preset": a.preset, "stage": a.stage}, path)

    net.train()
    t0 = time.time()
    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        total = 0.0
        for _ in range(a.grad_accum):
            if a.stage == "pretrain":
                x, y = get_pretrain_batch(bins["train"], a.batch_size, a.block_size, device)
            else:
                x, y = encode_cards([cards.card_to_text(sample(rng)) for _ in range(a.batch_size)],
                                    tok, bos, eos, pad, a.block_size, device)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                loss = net(x, y)[1] / a.grad_accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == 1:
            print(f"step {step}/{a.max_steps} | train loss {total:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.max_steps:
            v = evaluate()
            improved = v < best
            if improved:
                best, bad = v, 0
                save(best_path, final=True)
            else:
                bad += 1
            print(f"  >> val loss {v:.4f} (best {best:.4f}){' *saved best*' if improved else ''}", flush=True)
            if bad >= a.patience:
                print(f"early stop: val loss has not improved for {a.patience} evals (the model is starting to memorise)", flush=True)
                break
        if step % a.save_every == 0:
            save(ckpt_path)
    save(ckpt_path)
    print(f"done. best val loss {best:.4f}. best weights: {best_path}", flush=True)


if __name__ == "__main__":
    main()
