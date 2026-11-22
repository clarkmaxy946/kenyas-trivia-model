#!/usr/bin/env python3
"""
Human evaluation sheet (spec evaluation part (b)).

  python eval/rating_sheet.py --url http://127.0.0.1:8000 --cards 10 --out eval/ratings.csv

Asks the running service for cards and writes one row per word. Give the CSV to 3-5 Kenyan
players (or rate it yourselves). Columns to fill, per word:
  relevance      1-5   would a Kenyan audience recognise / value this word?
  alias_useful   1-5   is the one-word alias a fair, helpful clue?
  hint_accurate  1/0   is the hint factually true?
  hint_useful    1-5   does the hint teach the player something relevant?
Then:  python eval/rating_sheet.py --summarise eval/ratings.csv
"""
import argparse
import csv
import json
import statistics
import urllib.request

COLS = ["card", "word", "alias", "hint", "rater", "relevance", "alias_useful", "hint_accurate", "hint_useful"]


def summarise(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    for c in ("relevance", "alias_useful", "hint_accurate", "hint_useful"):
        vals = [float(r[c]) for r in rows if r.get(c, "").strip()]
        if vals:
            print(f"{c:>14}: mean {statistics.mean(vals):.2f}  (n={len(vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--cards", type=int, default=10)
    ap.add_argument("--out", default="eval/ratings.csv")
    ap.add_argument("--summarise")
    a = ap.parse_args()
    if a.summarise:
        return summarise(a.summarise)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i in range(a.cards):
            req = urllib.request.Request(a.url + "/generate_card", b"{}", {"Content-Type": "application/json"})
            card = json.loads(urllib.request.urlopen(req, timeout=300).read())
            for e in card["words"]:
                w.writerow({"card": i + 1, "word": e["word"], "alias": e["alias"], "hint": e["hint"]})
    print(f"wrote {a.out}: {a.cards} cards, {a.cards * 10} words to rate")


if __name__ == "__main__":
    main()
