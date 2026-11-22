#!/usr/bin/env python3
"""
Phase 2, step 2 (runs on your laptop): turn data/processed/entries_raw.jsonl
(from the Kaggle run) into clean entries that obey the Card Contract.

Checks, in order (first failure is the rejection reason):
  parse / hook / grounded   LLM said no, or output was not valid JSON
  word_length               target word longer than 5 words (hard to describe)
  alias_*                   alias not exactly one letters-only word, or it equals,
                            contains, or is contained in the target word
  hint_*                    hint not one English sentence of 6-30 words, or
                            talks about "the article/text", or has odd characters
  hint_*                    also: time-relative words (current, latest, recently)
  ungrounded_*              a number or capitalised name in the hint that does NOT
                            appear in the source article (cheap lexical fact guard)
  duplicate                 same target word already accepted

Each entry also carries views_60d (from data/processed/candidates.csv) so we can later keep only
well-known words. The audit CSV has two columns to fill: ok (1/0: hint true + usable clue) and
known (1/0: would an ordinary Kenyan at a party recognise this word?).

Outputs:
  data/processed/entries_filtered.jsonl   accepted entries (with source url)
  data/processed/entries_review.jsonl    valid entries set aside for YOUR decision, each with a
                                          `review_reason`: sensitive_topic (attacks/violence),
                                          time_sensitive ("is the president" goes stale),
                                          no_kenya_link (hint never mentions Kenya)
  data/processed/audit_sample.csv         random sample for YOUR hand check
  data/processed/rejections.csv           what was rejected and why (to tune rules)

After checking the audit sample (fill the `ok` column: 1 = correct, 0 = wrong):
  python scripts/04_filter_entries.py --audit-stats data/processed/audit_sample.csv
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "processed" / "entries_raw.jsonl"
ARTICLES = ROOT / "data" / "raw" / "articles.jsonl"
OUT = ROOT / "data" / "processed" / "entries_filtered.jsonl"
AUDIT = ROOT / "data" / "processed" / "audit_sample.csv"
REJ = ROOT / "data" / "processed" / "rejections.csv"
REVIEW = ROOT / "data" / "processed" / "entries_review.jsonl"
CANDIDATES = ROOT / "data" / "processed" / "candidates.csv"

MAX_WORD_WORDS = 5
HINT_MIN_WORDS, HINT_MAX_WORDS = 6, 25
AUDIT_SIZE = 150
ALWAYS_OK_CAPS = {"kenya", "kenyan", "kenyans", "african", "africa", "english", "swahili", "east"}

ALIAS_RE = re.compile(r"^[A-Za-z][a-z]+$")  # one word; rejects joined words like GamingCompany
HINT_BAD = re.compile(r"\b(the (article|text|passage)|according to|this (article|text)|current|currently|latest|newest|recently)\b", re.I)
SENSITIVE = re.compile(r"terror|attack|bomb|massacre|genocide|militant|extremis|murder|assassinat|\brape|torture|slaughter|abduct|hostage", re.I)
TIME_SENSITIVE = re.compile(r"\b(is|are|serves as|serving as)\s+(the\s+|a\s+)?(\w+\s+)?(president|prime minister|governor|speaker|chief justice|deputy president|senator|ceo|chairman|cabinet secretary)\b", re.I)
KENYA_LINK = re.compile(r"\b(kenya|kenyan|kenyans|nairobi|mombasa|kisumu|east africa|east african|mau mau|swahili|maasai|kikuyu|luo|kalenjin|kamba|luhya)\b", re.I)
HINT_CHARS = re.compile(r"^[A-Za-z0-9 ,.\-'’‘“”\"()&/:;%!?–—ĩũĨŨ]+$")


def tokens(s):
    return re.findall(r"[A-Za-z0-9ĩũĨŨ]+", s.lower())


def alias_problem(alias, word):
    if not ALIAS_RE.match(alias):
        return "alias_not_one_word"
    a, w = alias.lower(), word.lower()
    wt = tokens(word)
    if a == w or a in wt or a in w or (len(w) >= 4 and w in a):
        return "alias_overlaps_word"
    for t in wt:  # shared long prefix, e.g. Kenyatta / Kenyatta's
        if len(t) >= 5 and len(a) >= 5 and t[:5] == a[:5]:
            return "alias_overlaps_word"
    return None


def hint_problem(hint, word, article_lower, article_tokens):
    words = hint.split()
    if not (HINT_MIN_WORDS <= len(words) <= HINT_MAX_WORDS):
        return "hint_length"
    if not hint[0].isupper():
        return "hint_format"
    body = hint.rstrip()
    if not body.endswith((".", "!")):
        return "hint_not_sentence"
    inner = re.sub(r"\b(Mr|Mrs|Dr|St|Jr|Sr|Prof|Gen|Col|No)\.", "", body[:-1])
    inner = re.sub(r"\b[A-Z]\.", "", inner)  # initials
    inner = re.sub(r"\d\.\d", "", inner)
    if re.search(r"[.!?]\s+[A-Z]", inner) or "\n" in hint:
        return "hint_multi_sentence"
    if HINT_BAD.search(hint):
        return "hint_mentions_source"
    if not HINT_CHARS.match(hint):
        return "hint_odd_chars"
    for num in re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?![A-Za-z0-9])", hint):
        num = num.strip(",")
        if not re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", article_lower):
            return "ungrounded_number"
    wt = set(tokens(word))
    for i, w in enumerate(words):
        core = re.sub(r"(’s|'s)$", "", w.strip(".,;:!?()\"“”‘’'"))
        if i == 0 or not core or not core[0].isupper():
            continue
        parts = tokens(core)  # "Kenyan-American" -> kenyan, american; "Nyong'o" -> nyong, o
        if all(p in ALWAYS_OK_CAPS or p in wt or p in article_tokens for p in parts):
            continue
        return "ungrounded_name"
    return None


def load_views():
    """title -> pageviews in the last 60 days (a rough 'how well known is this' score)."""
    views = {}
    if CANDIDATES.exists():
        with open(CANDIDATES, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    views[r["title"]] = int(r["views_60d"])
                except (KeyError, ValueError):
                    pass
    return views


def audit_stats(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    done = [r for r in rows if r.get("ok", "").strip() in ("0", "1")]
    if not done:
        print("No rows with ok=0/1 yet.")
        return
    bad = sum(r["ok"].strip() == "0" for r in done)
    p = bad / len(done)
    se = (p * (1 - p) / len(done)) ** 0.5
    print(f"Checked {len(done)} of {len(rows)} sampled entries")
    print(f"Error rate: {p:.1%}  (about +/- {1.96 * se:.1%} at 95% confidence)")
    print("Record this number in the evaluation chapter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--articles", default=str(ARTICLES))
    ap.add_argument("--audit-stats", metavar="CSV", help="print error rate from a reviewed audit CSV")
    a = ap.parse_args()
    if a.audit_stats:
        audit_stats(a.audit_stats)
        return

    views = load_views()
    arts = {}
    with open(a.articles, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                arts[d["title"]] = d["text"]

    raw = []
    with open(a.raw, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))
    raw.sort(key=lambda r: r["title"])  # deterministic

    accepted, review, rejected, seen = [], [], [], set()
    reasons = Counter()
    for r in raw:
        why = None
        word = r["word"].strip()
        text = arts.get(r["title"], "")
        low = text.lower()
        toks = set(tokens(text))
        if not r.get("parsed"):
            why = "parse_failed"
        elif not r.get("hook"):
            why = "no_cultural_hook"
        elif r.get("grounded") is not True:
            why = "not_grounded"
        elif len(word.split()) > MAX_WORD_WORDS:
            why = "word_length"
        else:
            # try each candidate alias in order; keep the first one that passes
            cands = [c.strip() for c in (r.get("aliases") or [r.get("alias", "")]) if c.strip()] or [""]
            problems = [alias_problem(c, word) for c in cands]
            if all(problems):
                why = problems[0]
            else:
                r["alias"] = cands[problems.index(None)]
                why = hint_problem(r["hint"].strip(), word, low, toks)
        key = re.sub(r"\W+", "", word.lower())
        if not why and key in seen:
            why = "duplicate"
        if why:
            reasons[why] += 1
            rejected.append({"title": r["title"], "reason": why, "alias": r["alias"], "hint": r["hint"]})
            continue
        seen.add(key)
        h = r["hint"]
        if SENSITIVE.search(word + " " + h):
            review_reason = "sensitive_topic"
        elif TIME_SENSITIVE.search(h):
            review_reason = "time_sensitive"
        elif not KENYA_LINK.search(h):
            review_reason = "no_kenya_link"
        else:
            review_reason = ""
        entry = {
            "word": word, "alias": r["alias"].strip().capitalize(), "hint": r["hint"].strip(),
            "source_title": r["title"], "url": r["url"], "group": r["group"],
            "views_60d": views.get(r["title"], 0),
        }
        if review_reason:
            entry["review_reason"] = review_reason
            review.append(entry)
        else:
            accepted.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for e in accepted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(REVIEW, "w", encoding="utf-8") as f:
        for e in review:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(REJ, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["title", "reason", "alias", "hint"])
        w.writeheader()
        w.writerows(rejected)

    random.seed(42)
    sample = random.sample(accepted, min(AUDIT_SIZE, len(accepted)))
    with open(AUDIT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["word", "alias", "hint", "views_60d", "url", "source_intro", "ok", "known", "fix"])
        w.writeheader()
        for e in sample:
            intro = " ".join(arts.get(e["source_title"], "").split())[:500]
            w.writerow({"word": e["word"], "alias": e["alias"], "hint": e["hint"],
                        "views_60d": e["views_60d"], "url": e["url"],
                        "source_intro": intro, "ok": "", "known": "", "fix": ""})

    print(f"{len(raw)} raw | {len(accepted)} accepted | {len(review)} set aside for review | {len(rejected)} rejected")
    print("  review reasons:", dict(Counter(e["review_reason"] for e in review)))
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    print(f"By group (accepted): {dict(Counter(e['group'] for e in accepted))}")
    print("Accepted entries by popularity (60-day Wikipedia views):",
          {f">={t}": sum(e["views_60d"] >= t for e in accepted) for t in (300, 500, 1000, 2000)})
    print(f"Audit sample ({len(sample)}) -> {AUDIT}")


if __name__ == "__main__":
    main()
