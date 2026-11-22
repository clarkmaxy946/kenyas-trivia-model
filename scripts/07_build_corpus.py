#!/usr/bin/env python3
"""
Phase 3, step 1 (laptop): build everything the model trains on.

1. Split the 2,010 verified entries into train / val / test BY ENTRY (90/5/5, fixed by a hash of the
   word, so it never changes between runs). The val/test entries are never shown to the model as
   training cards, so val loss measures generalisation, not memory.
2. Pretraining text = the Wikipedia articles from Phase 1 (English text about Kenya, CC BY-SA).
   Articles behind val/test entries are LEFT OUT, so the held-out entries' facts stay unseen.
3. A fixed set of val cards (from val entries) for repeatable evaluation.

Outputs in data/corpus/:
  entries_train.jsonl  entries_val.jsonl  entries_test.jsonl   (each entry has serving: true/false)
  pretrain_train.txt   pretrain_val.txt
  val_cards.txt        (one card JSON per line)
Training cards are NOT written to disk: train/train.py draws fresh random cards every step.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import cards  # noqa: E402

P = ROOT / "data" / "processed"
OUT = ROOT / "data" / "corpus"
VAL_CARDS = 300


def bucket(word):
    h = int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def clean_article(text):
    text = re.sub(r"={2,}\s*(.+?)\s*={2,}", r"\n\1.\n", text)  # "== History ==" -> "History."
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    serving = {json.loads(l)["word"] for l in open(P / "entries_serving.jsonl", encoding="utf-8") if l.strip()}
    entries = [json.loads(l) for l in open(P / "entries_all.jsonl", encoding="utf-8") if l.strip()]
    split = {"train": [], "val": [], "test": []}
    for e in entries:
        e["serving"] = e["word"] in serving
        split[bucket(e["word"])].append(e)
    for name, items in split.items():
        with open(OUT / f"entries_{name}.jsonl", "w", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"entries_{name}: {len(items)}  serving={sum(e['serving'] for e in items)}  "
              f"groups={dict(Counter(e['group'] for e in items))}")

    held_out_titles = {e["source_title"] for n in ("val", "test") for e in split[n]}
    n_art = n_chars = 0
    tr = open(OUT / "pretrain_train.txt", "w", encoding="utf-8")
    va = open(OUT / "pretrain_val.txt", "w", encoding="utf-8")
    for line in open(ROOT / "data" / "raw" / "articles.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d["title"] in held_out_titles:
            continue
        txt = clean_article(d["text"]) + "\n<|eos|>\n"
        h = int(hashlib.md5(d["title"].encode("utf-8")).hexdigest(), 16) % 100
        (va if h < 2 else tr).write(txt)
        n_art += 1
        n_chars += len(txt)
    tr.close()
    va.close()
    print(f"pretraining text: {n_art} articles, {n_chars / 1e6:.1f}M characters "
          f"({len(held_out_titles)} held-out articles excluded)")

    sample = cards.make_sampler(split["val"])
    rng = cards.rng_from(1234)
    with open(OUT / "val_cards.txt", "w", encoding="utf-8") as f:
        for _ in range(VAL_CARDS):
            f.write(cards.card_to_text(sample(rng)) + "\n")
    print(f"val_cards.txt: {VAL_CARDS} cards")


if __name__ == "__main__":
    main()
