#!/usr/bin/env python3
"""
Generate cards with a trained model and measure how trustworthy they are.

  python train/generate.py --ckpt run1/best_cards.pt --data data/corpus --n 50

Report (the numbers for the evaluation chapter):
  valid cards     cards that meet every Card Contract rule (10 words, one-word aliases, one-sentence hints)
  exact entries   words copied exactly from a verified entry (word + alias + hint all match)
  altered         the word exists in the dataset but the alias/hint was changed  -> risk of wrong facts
  novel words     words that are not in the dataset at all                       -> invented by the model
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cards  # noqa: E402
import model as M  # noqa: E402

PREFIX = '{"words":[{"word":"'


def load(ckpt, tokenizer, device):
    ck = torch.load(ckpt, map_location=device)
    net = M.GPT(M.GPTConfig(**ck["config"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, Tokenizer.from_file(str(tokenizer))


def generate_card_text(net, tok, device, temperature=0.8, top_k=40, max_new=480):
    bos, eos = tok.token_to_id(cards.BOS), tok.token_to_id(cards.EOS)
    ids = [bos] + tok.encode(PREFIX).ids  # the output format is fixed, so we start it for the model
    idx = torch.tensor([ids], device=device)
    out = net.generate(idx, max_new, eos, temperature, top_k)[0].tolist()
    if out and out[-1] == eos:
        out = out[:-1]
    return tok.decode(out[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="data/corpus folder (for the verified-entry check)")
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer.json"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--show", type=int, default=1, help="print this many generated cards")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, tok = load(a.ckpt, a.tokenizer, device)
    verified = {}
    for name in ("train", "val", "test"):
        for l in open(Path(a.data) / f"entries_{name}.jsonl", encoding="utf-8"):
            if l.strip():
                e = json.loads(l)
                verified[cards.norm(e["word"])] = e

    valid = exact = altered = novel = total = 0
    problems = {}
    for i in range(a.n):
        text = generate_card_text(net, tok, device, a.temperature, a.top_k)
        card, probs = cards.check_card(text)
        for p in probs:
            problems[p] = problems.get(p, 0) + 1
        if i < a.show:
            print(text, "\n")
        if card is None:
            continue
        valid += 1
        for w in card["words"]:
            total += 1
            e = verified.get(cards.norm(w["word"]))
            if e is None:
                novel += 1
            elif (e["alias"], e["hint"]) == (w["alias"], w["hint"]):
                exact += 1
            else:
                altered += 1
    print(f"cards generated: {a.n} | valid under the Card Contract: {valid} ({100 * valid / a.n:.0f}%)")
    if total:
        print(f"words in valid cards: {total} | exact verified entries {100 * exact / total:.0f}% | "
              f"altered {100 * altered / total:.0f}% | novel words {100 * novel / total:.0f}%")
    if problems:
        print("contract problems:", dict(sorted(problems.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
