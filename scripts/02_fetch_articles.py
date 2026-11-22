#!/usr/bin/env python3
"""
Phase 1, step 2: download the full text of every entry you marked keep=1.

Input : data/processed/candidates.csv   (after you filled the `keep` column)
Output: data/raw/articles.jsonl         (one JSON object per line: title, url, group, text)

Resumable: titles already in articles.jsonl are skipped on re-run.
Wikipedia text is CC BY-SA 4.0; `url` is kept for attribution (see SOURCES.md).
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN_CSV = ROOT / "data" / "processed" / "candidates.csv"
OUT = ROOT / "data" / "raw" / "articles.jsonl"
MIN_CHARS = 500  # skip stubs: too little text to ground a hint


def load_helpers():
    """Reuse api_get and the contact check from script 01."""
    spec = importlib.util.spec_from_file_location("collect", HERE / "01_collect_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    collect = load_helpers()
    if "YOUR-EMAIL" in collect.CONTACT:
        print("Edit CONTACT in scripts/01_collect_candidates.py first.")
        sys.exit(1)

    with open(IN_CSV, newline="", encoding="utf-8-sig") as f:
        kept = [r for r in csv.DictReader(f) if r["keep"].strip() == "1"]
    if not kept:
        print("No rows with keep=1 in candidates.csv. Fill the `keep` column first.")
        sys.exit(1)

    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["title"] for line in f if line.strip()}

    todo = [r for r in kept if r["title"] not in done]
    print(f"{len(kept)} kept, {len(done)} already downloaded, {len(todo)} to go")

    skipped = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for r in tqdm(todo, unit="article"):
            data = collect.api_get({
                "action": "query", "titles": r["title"], "redirects": "1",
                "prop": "extracts", "explaintext": "1", "exlimit": "1",
            })
            pages = data.get("query", {}).get("pages", [])
            text = (pages[0].get("extract") or "").strip() if pages else ""
            if len(text) < MIN_CHARS:
                skipped += 1
                continue
            out.write(json.dumps({
                "title": r["title"], "url": r["url"], "group": r["group"], "text": text,
            }, ensure_ascii=False) + "\n")
            out.flush()
    print(f"Done. Skipped {skipped} stubs (< {MIN_CHARS} chars). Output: {OUT}")


if __name__ == "__main__":
    main()
