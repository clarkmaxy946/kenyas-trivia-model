#!/usr/bin/env python3
"""
Phase 2, step 4 (laptop): build the final verified dataset.

serving tier  = accepted entries with >= --min-views (default 500) views
                + rescued entries from rescue_sheet_filled.csv marked keep
all tier      = every accepted entry + rescued entries (used later for model training)

Every entry is re-validated against the Card Contract (one-word alias that is not the word,
one-sentence English hint) so a typo in a hand fix can't get through.

Reads : data/processed/entries_filtered.jsonl, rescue_sheet_filled.csv, entries_raw.jsonl
Writes: data/processed/entries_serving.jsonl, entries_all.jsonl, build_problems.csv
Usage : python scripts/06_build_serving.py [--min-views 500] [--sheet data/processed/rescue_sheet_filled.csv]
"""
import argparse
import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"

spec = importlib.util.spec_from_file_location("filt", ROOT / "scripts" / "04_filter_entries.py")
filt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filt)


def contract_problem(e):
    p = filt.alias_problem(e["alias"], e["word"])
    if p:
        return p
    hint = e["hint"].strip()
    n = len(hint.split())
    if not (filt.HINT_MIN_WORDS <= n <= filt.HINT_MAX_WORDS):
        return "hint_length"
    if not hint[0].isupper() or not hint.endswith((".", "!")):
        return "hint_format"
    inner = re.sub(r"\b(Mr|Mrs|Dr|St|Jr|Sr|Prof|Gen|Col|No)\.", "", hint[:-1])
    inner = re.sub(r"\b[A-Z]\.", "", inner)
    if re.search(r"[.!?]\s+[A-Z]", inner):
        return "hint_multi_sentence"
    if filt.HINT_BAD.search(hint):
        return "hint_mentions_source"
    if not re.match(r"^[A-Za-z0-9 ,.\-'’\"()&/:;%!?–—ĩũĨŨ]+$", hint):
        return "hint_odd_chars"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-views", type=int, default=500)
    ap.add_argument("--sheet", default=str(P / "rescue_sheet_filled.csv"))
    a = ap.parse_args()

    meta = {}
    for line in open(P / "entries_raw.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            meta[r["title"]] = (r["url"], r["group"])
    views = filt.load_views()

    entries = [json.loads(l) for l in open(P / "entries_filtered.jsonl", encoding="utf-8") if l.strip()]
    for e in entries:
        e["source"] = "accepted"
    for r in csv.DictReader(open(a.sheet, newline="", encoding="utf-8-sig")):
        if r["decision"].strip().lower() != "keep":
            continue
        url, group = meta.get(r["title"], ("", ""))
        entries.append({
            "word": r["word"].strip(), "alias": (r["fix_alias"].strip() or r["alias"].strip()).capitalize(),
            "hint": r["fix_hint"].strip() or r["hint"].strip(), "source_title": r["title"], "url": url,
            "group": group, "views_60d": views.get(r["title"], 0), "source": "rescued",
        })

    good, problems, seen = [], [], set()
    for e in entries:
        key = re.sub(r"\W+", "", e["word"].lower())
        why = contract_problem(e) or ("duplicate" if key in seen else None)
        if why:
            problems.append({"word": e["word"], "problem": why, "alias": e["alias"], "hint": e["hint"]})
            continue
        seen.add(key)
        good.append(e)

    serving = [e for e in good if e["views_60d"] >= a.min_views or e["source"] == "rescued"]
    for name, data in (("entries_all.jsonl", good), ("entries_serving.jsonl", serving)):
        with open(P / name, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(P / "build_problems.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["word", "problem", "alias", "hint"])
        w.writeheader()
        w.writerows(problems)
    print(f"all tier: {len(good)} | serving tier (>= {a.min_views} views or rescued): {len(serving)} | problems: {len(problems)}")
    print("serving by group:", dict(Counter(e["group"] for e in serving)))
    print("serving by source:", dict(Counter(e["source"] for e in serving)))
    if problems:
        print("problems:", dict(Counter(p["problem"] for p in problems)), "-> build_problems.csv")


if __name__ == "__main__":
    main()
