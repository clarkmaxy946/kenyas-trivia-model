# Kenya's Trivia: Model Workstream

Small GPT-style model, trained from scratch, that generates Kenya's Trivia cards.
Contract: see Section 2 of `Kenyas_Trivia_Model_Specification.docx`
(a card = exactly 10 x {word, alias, hint}; alias = one word; hint = one English sentence).

## Layout
- `scripts/`  data pipeline (numbered in run order)
- `data/raw/` downloaded source material (never edited by hand)
- `data/processed/` generated files (candidate lists, cards, corpora)
- `train/`    model + training code (Phase 4)
- `service/`  FastAPI card service (Phase 5)
- `eval/`     latency + human-rating tools (Phase 6)
- `SOURCES.md` licence/provenance log (keep it updated)

## Phase 1 run order
1. `python scripts/01_collect_candidates.py`  -> data/processed/candidates.csv
2. Open candidates.csv, fill the `keep` column (1 = has a cultural hook, 0 = drop)
3. `python scripts/02_fetch_articles.py`      -> data/raw/articles.jsonl (full text of kept entries)

## Rules carried over from the spec
No Sheng. No forbidden words. No categories on cards. Hints in English.
Every entry keeps its source URL.
