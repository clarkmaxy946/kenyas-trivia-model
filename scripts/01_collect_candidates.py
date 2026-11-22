#!/usr/bin/env python3
"""
Phase 1, step 1: collect candidate entries for Kenya's Trivia from English Wikipedia.

What it does
  1. Starts from seed categories (history, people, culture, places, ...).
  2. Walks subcategories, but only into ones that look Kenyan (see KENYA_TOKENS),
     so the crawl does not drift into unrelated topics.
  3. Collects article titles, then fetches for each: URL, 3-sentence intro,
     and page views over the last ~60 days (popularity = a rough proxy for
     "a Kenyan today would still recognise this", i.e. the cultural-hook rule).
  4. Writes data/processed/candidates.csv, sorted by popularity, with an empty
     `keep` column for YOU to fill in (1 = keep, 0 = drop).

Run:  python scripts/01_collect_candidates.py
Needs internet access to en.wikipedia.org. Re-running reuses the cached crawl
(data/raw/category_crawl.json); delete that file to crawl again.

Wikipedia text is CC BY-SA 4.0: keep the `url` column, it is the attribution.
"""
import csv
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import requests
from tqdm import tqdm

# ----------------------------------------------------------------- config ---
# Wikimedia asks for a User-Agent that says who you are. Put your own contact.
CONTACT = "clarkmaxwell946@gmail.com"
USER_AGENT = f"KenyasTriviaDataBuilder/0.1 (DeKUT student project; {CONTACT})"

API = "https://en.wikipedia.org/w/api.php"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
CRAWL_CACHE = RAW_DIR / "category_crawl.json"
OUT_CSV = OUT_DIR / "candidates.csv"

MAX_DEPTH = 3          # how many subcategory levels to descend from each seed
MAX_PAGES = 40000      # safety cap on collected titles
MIN_VIEWS_60D = 100    # drop very obscure pages (~<2 views/day). Lower to keep more.
MAX_ROWS = 8000        # rows written to the review CSV (most popular first)
SLEEP = 0.15           # polite pause between API calls

# seed category -> group label (used only as a hint in the review sheet)
SEEDS = {
    # history and events
    "Category:History of Kenya": "history",
    "Category:Mau Mau Uprising": "history",
    "Category:Kenyan independence movement": "history",
    # people
    "Category:Kenyan people": "people",
    "Category:Kenyan politicians": "people",
    "Category:Kenyan sportspeople": "people",
    "Category:Kenyan musicians": "people",
    "Category:Kenyan writers": "people",
    "Category:Kenyan businesspeople": "people",
    "Category:Kenyan activists": "people",
    # culture
    "Category:Kenyan culture": "culture",
    "Category:Kenyan cuisine": "culture",
    "Category:Public holidays in Kenya": "culture",
    "Category:National symbols of Kenya": "culture",
    "Category:Kenyan music": "culture",
    "Category:Kenyan literature": "culture",
    "Category:Ethnic groups in Kenya": "culture",
    "Category:Kenyan television": "culture",
    "Category:Kenyan films": "culture",
    # places and things
    "Category:Tourist attractions in Kenya": "places",
    "Category:Mountains of Kenya": "places",
    "Category:National parks of Kenya": "places",
    "Category:Geography of Kenya": "places",
    "Category:Companies of Kenya": "things",
    "Category:Sport in Kenya": "things",
}

# Only descend into a subcategory whose name contains one of these.
KENYA_TOKENS = (
    "kenya", "mau mau", "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
    "kikuyu", "luo ", "luhya", "kalenjin", "kamba", "maasai", "turkana",
    "mijikenda", "swahili coast", "rift valley", "lake victoria",
)

# Titles that are never a single guessable "word".
SKIP_TITLE = re.compile(
    r"^(List of|Lists of|Outline of|Index of|Timeline of|Template:|Category:|"
    r"Wikipedia:|File:|Portal:)|\(disambiguation\)$|\bin the \d{4}\b|^\d{4} in ",
    re.IGNORECASE,
)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


# ---------------------------------------------------------------- helpers ---
def api_get(params, retries=5):
    """GET the MediaWiki API with retry/backoff. Returns parsed JSON."""
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(retries):
        try:
            r = _session.get(API, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if "error" in data and data["error"].get("code") == "maxlag":
                    time.sleep(5)
                    continue
                time.sleep(SLEEP)
                return data
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  network issue ({e}); retrying...", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("API failed after retries")


def category_members(cat_title):
    """Yield (title, is_category) for the members of one category."""
    params = {
        "action": "query", "list": "categorymembers", "cmtitle": cat_title,
        "cmtype": "page|subcat", "cmnamespace": "0|14", "cmlimit": "max",
    }
    while True:
        data = api_get(params)
        if "error" in data:
            print(f"  skipped {cat_title}: {data['error'].get('info')}", file=sys.stderr)
            return
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"], m["ns"] == 14
        if "continue" in data:
            params.update(data["continue"])
        else:
            return


def looks_kenyan(cat_title):
    t = cat_title.lower()
    return any(tok in t for tok in KENYA_TOKENS)


def crawl():
    """Breadth-first walk of the seed categories. Returns {title: seed_group}."""
    pages = {}
    seen_cats = set()
    queue = deque((seed, 0, group) for seed, group in SEEDS.items())
    pbar = tqdm(desc="categories crawled", unit="cat")
    while queue and len(pages) < MAX_PAGES:
        cat, depth, group = queue.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        pbar.update(1)
        for title, is_cat in category_members(cat):
            if is_cat:
                if depth < MAX_DEPTH and looks_kenyan(title):
                    queue.append((title, depth + 1, group))
            elif not SKIP_TITLE.search(title) and title not in pages:
                pages[title] = group
    pbar.close()
    return pages


def fetch_details(titles):
    """Fetch URL, intro text and 60-day page views for titles. Returns {title: dict}."""
    out = {}
    origin = {}  # final title -> the title we asked for (to keep the group label)
    batch = 20  # the extracts API returns intro text for at most 20 titles per call
    for i in tqdm(range(0, len(titles), batch), desc="details", unit="batch"):
        chunk = titles[i:i + batch]
        params = {
            "action": "query", "titles": "|".join(chunk), "redirects": "1",
            "prop": "extracts|pageviews|info", "inprop": "url",
            "exintro": "1", "explaintext": "1", "exsentences": "3", "exlimit": "max",
            "pvipdays": "60",
        }
        merged = {}
        hops = {}
        while True:
            data = api_get(params)
            for m in data.get("query", {}).get("normalized", []) + data.get("query", {}).get("redirects", []):
                hops[m["from"]] = m["to"]
            for p in data.get("query", {}).get("pages", []):
                cur = merged.setdefault(p["title"], {"views": {}})
                if "extract" in p:
                    cur["extract"] = p["extract"]
                if "fullurl" in p:
                    cur["url"] = p["fullurl"]
                if p.get("pageviews"):
                    cur["views"].update(p["pageviews"])
                if p.get("missing"):
                    cur["missing"] = True
            if "continue" in data:
                params.update(data["continue"])
            else:
                break
        for asked in chunk:  # follow normalisation/redirect chains to the final title
            final, steps = asked, 0
            while final in hops and steps < 5:
                final, steps = hops[final], steps + 1
            origin.setdefault(final, asked)
        for title, d in merged.items():
            if d.get("missing"):
                continue
            views = sum(v for v in d["views"].values() if v)
            out[title] = {
                "url": d.get("url", ""),
                "intro": " ".join((d.get("extract") or "").split()),
                "views_60d": views,
            }
    return out, origin


# ------------------------------------------------------------------- main ---
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "YOUR-EMAIL" in CONTACT:
        print("Edit CONTACT at the top of this file first (Wikimedia asks for it).")
        sys.exit(1)

    if CRAWL_CACHE.exists():
        pages = json.loads(CRAWL_CACHE.read_text(encoding="utf-8"))
        print(f"Loaded cached crawl: {len(pages)} titles")
    else:
        pages = crawl()
        CRAWL_CACHE.write_text(json.dumps(pages, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"Crawl done: {len(pages)} titles")

    details, origin = fetch_details(sorted(pages))

    rows = []
    for title, d in details.items():
        if d["views_60d"] < MIN_VIEWS_60D or not d["intro"]:
            continue
        group = pages.get(origin.get(title, title), "")
        rows.append({
            "title": title, "group": group, "views_60d": d["views_60d"],
            "intro": d["intro"], "url": d["url"], "keep": "",
        })
    rows.sort(key=lambda r: r["views_60d"], reverse=True)
    rows = rows[:MAX_ROWS]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["title", "group", "views_60d", "intro", "url", "keep"])
        w.writeheader()
        w.writerows(rows)

    by_group = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    print(f"\nWrote {len(rows)} candidates to {OUT_CSV}")
    print("By group:", by_group)
    print("Next: open the CSV, fill `keep` with 1 or 0 (see README), then run 02_fetch_articles.py")


if __name__ == "__main__":
    main()
