#!/usr/bin/env python3
"""
Held-out generation test: can the model write its OWN alias and hint for words it never trained on?

Put this file in  eval/heldout_test.py  and run it from the project root.

STEP 1 - generate + auto-check + make a blind rating sheet
  python eval/heldout_test.py --ckpt model/best_cards_fp16.pt --data data/corpus --n 40 --control

STEP 2 - give eval/heldout_ratings.csv to 3-5 raters (they do NOT see which row is the model's).
         Keep eval/heldout_key.csv to yourselves.

STEP 3 - summarise the filled sheet
  python eval/heldout_test.py --summarise eval/heldout_ratings.csv --key eval/heldout_key.csv

How it works
  The model was trained on cards shaped  {"words":[{"word":..,"alias":..,"hint":..},...]}.
  For each held-out word (from entries_val / entries_test, never used as training cards, and whose own
  Wikipedia article was left out of pretraining) we start a card with that word and let the model
  continue: it must write the alias and the hint itself. Nothing is looked up from the dataset.

  Rows in the rating sheet, shuffled together so raters cannot tell them apart:
    model    the alias + hint the model wrote
    verified the dataset's alias + hint for the same word (the reference)
    control  (--control) a verified hint belonging to a DIFFERENT word; raters should mark it wrong,
             which shows the raters are paying attention
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import cards  # noqa: E402

OUT_RE = re.compile(r'^([^"]*)","hint":"((?:[^"\\]|\\.)*)"')


def load_entries(data, names):
    items = []
    for n in names:
        for line in open(Path(data) / f"entries_{n}.jsonl", encoding="utf-8"):
            if line.strip():
                items.append(json.loads(line))
    return items


def write_alias_hint(net, tok, device, word, temperature, top_k, max_new=90):
    """Start a card with `word` and let the model write the alias and hint. Returns (alias, hint) or None."""
    import torch
    prompt = '{"words":[{"word":"' + word.replace("\\", "\\\\").replace('"', '\\"') + '","alias":"'
    bos, eos = tok.token_to_id(cards.BOS), tok.token_to_id(cards.EOS)
    ids = [bos] + tok.encode(prompt).ids
    idx = torch.tensor([ids], device=device)
    out = net.generate(idx, max_new, eos, temperature, top_k)[0].tolist()
    text = tok.decode(out[1:])
    rest = text[len(prompt):] if text.startswith(prompt) else text.split('"alias":"', 1)[-1]
    m = OUT_RE.match(rest)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def auto_checks(word, alias, hint):
    """The same per-word rules as the Card Contract (train/cards.py)."""
    problems = []
    if not cards.ALIAS_RE.match(alias):
        problems.append("alias_not_one_word")
    elif cards.norm(alias) in cards.norm(word):
        problems.append("alias_overlaps_word")
    n = len(hint.split())
    if not (6 <= n <= 25):
        problems.append("hint_length")
    if not hint.endswith((".", "!")) or not hint[:1].isupper():
        problems.append("hint_format")
    if re.search(r"[.!?]\s+[A-Z]", re.sub(r"\b(Mr|Mrs|Dr|St|Jr|Sr|Prof)\.|\b[A-Z]\.", "", hint[:-1])):
        problems.append("hint_multi_sentence")
    return problems


def jaccard(a, b):
    a, b = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(a & b) / len(a | b) if a | b else 0.0


def summarise(ratings, key):
    src = {r["item"]: r["source"] for r in csv.DictReader(open(key, newline="", encoding="utf-8-sig"))}
    by = {}
    for r in csv.DictReader(open(ratings, newline="", encoding="utf-8-sig")):
        by.setdefault(src.get(r["item"], "?"), []).append(r)
    print(f"{'source':>9} {'rows':>5} {'hint accurate':>14} {'hint useful (1-5)':>18} {'alias useful (1-5)':>19}")
    for s in ("model", "verified", "control"):
        rows = by.get(s, [])
        if not rows:
            continue

        def mean(c):
            v = [float(r[c]) for r in rows if r.get(c, "").strip()]
            return f"{statistics.mean(v):.2f}" if v else "-"

        acc = [float(r["hint_accurate"]) for r in rows if r.get("hint_accurate", "").strip()]
        a = f"{100 * statistics.mean(acc):.0f}% (n={len(acc)})" if acc else "-"
        print(f"{s:>9} {len(rows):>5} {a:>14} {mean('hint_useful'):>18} {mean('alias_useful'):>19}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default="data/corpus")
    ap.add_argument("--tokenizer", default=str(HERE.parent / "train" / "tokenizer.json"))
    ap.add_argument("--n", type=int, default=40, help="number of held-out words to test")
    ap.add_argument("--all", action="store_true", help="include held-out words that are not serving-tier")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--control", action="store_true", help="add mismatched-hint control rows")
    ap.add_argument("--out", default=str(HERE / "heldout_ratings.csv"))
    ap.add_argument("--key", default=str(HERE / "heldout_key.csv"))
    ap.add_argument("--summarise", help="summarise a filled rating sheet (needs --key)")
    a = ap.parse_args()

    if a.summarise:
        return summarise(a.summarise, a.key)
    if not a.ckpt:
        ap.error("--ckpt is required")

    import torch
    import generate as G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, tok = G.load(a.ckpt, a.tokenizer, device)

    held = load_entries(a.data, ("val", "test"))
    train = load_entries(a.data, ("train",))
    train_hints = {e["hint"].strip().lower() for e in train}
    pool = held if a.all else [e for e in held if e.get("serving")]
    rng = random.Random(a.seed)
    sample = rng.sample(pool, min(a.n, len(pool)))
    print(f"held-out entries: {len(held)} (serving-tier {sum(1 for e in held if e.get('serving'))}); testing {len(sample)}")

    rows, key = [], []
    n_parsed = n_clean = n_copy = n_alias_same = 0
    overlaps, probs_count = [], {}
    for e in sample:
        got = write_alias_hint(net, tok, device, e["word"], a.temperature, a.top_k)
        if got is None:
            probs_count["unparseable"] = probs_count.get("unparseable", 0) + 1
            alias, hint = "", ""
        else:
            alias, hint = got
            n_parsed += 1
            probs = auto_checks(e["word"], alias, hint)
            for p in probs:
                probs_count[p] = probs_count.get(p, 0) + 1
            n_clean += not probs
            n_copy += hint.lower() in train_hints  # copied a training hint word-for-word?
            n_alias_same += cards.norm(alias) == cards.norm(e["alias"])
            overlaps.append(jaccard(hint, e["hint"]))
        if got is not None:
            rows.append((e["word"], "model", alias, hint))
        rows.append((e["word"], "verified", e["alias"], e["hint"]))
        if a.control:
            other = rng.choice([x for x in held if x["word"] != e["word"]])
            rows.append((e["word"], "control", other["alias"], other["hint"]))

    rng.shuffle(rows)
    cols = ["item", "word", "alias", "hint", "rater", "hint_accurate", "hint_useful", "alias_useful"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f, open(a.key, "w", newline="", encoding="utf-8-sig") as g:
        w, k = csv.DictWriter(f, fieldnames=cols), csv.writer(g)
        w.writeheader()
        k.writerow(["item", "word", "source"])
        for i, (word, source, alias, hint) in enumerate(rows, 1):
            w.writerow({"item": i, "word": word, "alias": alias, "hint": hint})
            k.writerow([i, word, source])

    n = len(sample)
    print(f"\nAUTOMATIC RESULTS (temperature {a.temperature}, {n} held-out words)")
    print(f"  model produced a parseable alias+hint     {n_parsed}/{n}")
    print(f"  ...that also pass the Contract rules       {n_clean}/{n}")
    print(f"  hint copied word-for-word from training    {n_copy}/{n_parsed or 1}   (high = memorising, not writing)")
    print(f"  alias identical to the verified alias      {n_alias_same}/{n_parsed or 1}")
    if overlaps:
        print(f"  mean word overlap with the verified hint   {statistics.mean(overlaps):.2f}  (0 = none, 1 = identical)")
    if probs_count:
        print(f"  problems: {dict(sorted(probs_count.items(), key=lambda kv: -kv[1]))}")
    print(f"\nrating sheet: {a.out}  ({len(rows)} rows)\nkey (keep private): {a.key}")
    print("Raters fill: hint_accurate (1/0), hint_useful (1-5), alias_useful (1-5), rater (name).")


if __name__ == "__main__":
    main()
