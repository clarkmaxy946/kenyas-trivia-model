"""
Shared card logic: how a card is written as text, how cards are sampled from the verified
entries, and how a generated card is checked against the Card Contract (spec Section 2).
Used by scripts/07_build_corpus.py, train/train.py and train/generate.py.
"""
import json
import random
import re

BOS, EOS, PAD = "<|bos|>", "<|eos|>", "<|pad|>"
CARD_SIZE = 10

# How often a card slot is drawn from each group. People are ~57% of the data, so left alone
# every card would be mostly people. Slots are filled without replacement, so a card never repeats a word.
GROUP_P = {"people": 0.35, "places": 0.20, "history": 0.15, "culture": 0.15, "things": 0.15}
SERVING_WEIGHT = 3.0  # well-known (serving-tier) entries are drawn this many times more often

ALIAS_RE = re.compile(r"^[A-Za-z][a-z]+$")


def card_to_text(entries):
    """The exact JSON the API returns, written compactly. Key order is fixed."""
    words = [{"word": e["word"], "alias": e["alias"], "hint": e["hint"]} for e in entries]
    return json.dumps({"words": words}, ensure_ascii=False, separators=(",", ":"))


def make_sampler(entries):
    by_group = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)
    groups = [g for g in GROUP_P if by_group.get(g)]
    probs = [GROUP_P[g] for g in groups]

    def sample(rng):
        chosen, used = [], set()
        guard = 0
        while len(chosen) < CARD_SIZE and guard < 1000:
            guard += 1
            g = rng.choices(groups, probs)[0]
            pool = by_group[g]
            w = [SERVING_WEIGHT if e.get("serving") else 1.0 for e in pool]
            e = rng.choices(pool, w)[0]
            if e["word"] in used:
                continue
            used.add(e["word"])
            chosen.append(e)
        return chosen

    return sample


def norm(s):
    return re.sub(r"\W+", "", s.lower())


def check_card(text):
    """Return (card_or_None, problems). A card passes only if it meets every Contract rule."""
    problems = []
    try:
        obj = json.loads(text)
    except Exception:
        return None, ["not_valid_json"]
    words = obj.get("words") if isinstance(obj, dict) else None
    if not isinstance(words, list):
        return None, ["no_words_list"]
    if len(words) != CARD_SIZE:
        problems.append(f"count_{len(words)}")
    seen = set()
    for i, w in enumerate(words):
        if not isinstance(w, dict) or set(w) != {"word", "alias", "hint"} or not all(isinstance(v, str) for v in w.values()):
            problems.append("bad_keys")
            continue
        word, alias, hint = w["word"].strip(), w["alias"].strip(), w["hint"].strip()
        if not word:
            problems.append("empty_word")
        if not ALIAS_RE.match(alias):
            problems.append("alias_not_one_word")
        elif norm(alias) == norm(word) or norm(alias) in norm(word).split() or norm(alias) in norm(word):
            problems.append("alias_overlaps_word")
        n = len(hint.split())
        if not (6 <= n <= 25):
            problems.append("hint_length")
        if not hint.endswith((".", "!")) or not hint[:1].isupper():
            problems.append("hint_format")
        if re.search(r"[.!?]\s+[A-Z]", re.sub(r"\b(Mr|Mrs|Dr|St|Jr|Sr|Prof)\.|\b[A-Z]\.", "", hint[:-1])):
            problems.append("hint_multi_sentence")
        if norm(word) in seen:
            problems.append("duplicate_word")
        seen.add(norm(word))
    return (obj if not problems else None), problems


def rng_from(seed):
    return random.Random(seed)
