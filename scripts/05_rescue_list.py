#!/usr/bin/env python3
"""
Phase 2, step 3 (laptop): list WELL-KNOWN entries that the filter threw out or set aside,
so you can rescue the good ones by hand. Famous words matter most for the game.

Reads : data/processed/rejections.csv, entries_review.jsonl, candidates.csv, data/raw/articles.jsonl
Writes: data/processed/rescue_sheet.csv   (open in Excel, fill `decision`)

decision column:  keep  = the hint is fine as is (or you fixed it in `fix_hint`)
                  drop  = leave it out
Rows from entries_review.jsonl come with a ready hint+alias; rows from rejections.csv have the
model's rejected hint and alias, so check them against `source_intro` before saying keep.

Usage:  python scripts/05_rescue_list.py [--min-views 500]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
# reasons worth a human look (not parse failures, duplicates or no cultural hook)
RESCUABLE = {"not_grounded", "ungrounded_name", "ungrounded_number", "alias_not_one_word",
             "alias_overlaps_word", "hint_odd_chars", "hint_length", "hint_format",
             "hint_not_sentence", "hint_multi_sentence", "hint_mentions_source"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-views", type=int, default=500)
    a = ap.parse_args()

    views = {}
    with open(P / "candidates.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                views[r["title"]] = int(r["views_60d"])
            except (KeyError, ValueError):
                pass
    intro = {}
    with open(ROOT / "data" / "raw" / "articles.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                intro[d["title"]] = " ".join(d["text"].split())[:500]

    out = []
    review_path = P / "entries_review.jsonl"
    if review_path.exists():
        for line in open(review_path, encoding="utf-8"):
            if not line.strip():
                continue
            e = json.loads(line)
            v = e.get("views_60d") or views.get(e["source_title"], 0)
            if v >= a.min_views:
                out.append({"title": e["source_title"], "views_60d": v, "why": "review:" + e["review_reason"],
                            "word": e["word"], "alias": e["alias"], "hint": e["hint"]})
    with open(P / "rejections.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = views.get(r["title"], 0)
            if r["reason"] in RESCUABLE and v >= a.min_views:
                out.append({"title": r["title"], "views_60d": v, "why": "rejected:" + r["reason"],
                            "word": r["title"], "alias": r["alias"], "hint": r["hint"]})
    out.sort(key=lambda x: -x["views_60d"])
    cols = ["title", "views_60d", "why", "word", "alias", "hint", "source_intro", "decision", "fix_alias", "fix_hint"]
    with open(P / "rescue_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in out:
            w.writerow({**x, "source_intro": intro.get(x["title"], ""), "decision": "", "fix_alias": "", "fix_hint": ""})
    print(f"{len(out)} well-known entries (>= {a.min_views} views) to look at -> {P / 'rescue_sheet.csv'}")


if __name__ == "__main__":
    main()
