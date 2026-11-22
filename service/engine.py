"""
Card engine with the safety net.

The model proposes cards. Every word it proposes is looked up in the VERIFIED dataset; the alias
and hint that are served always come from the verified entry, never from the model's own text.
Words the model invents (not in the dataset) are discarded. If the model cannot supply 10 usable
words (too many invented, or already used in the lobby), the gap is filled from the verified pool.
So the accuracy of every served card equals the measured accuracy of the dataset.
"""
import json
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import cards  # noqa: E402

WORD_RE = re.compile(r'"word":"((?:[^"\\]|\\.)*)"')


class Verified:
    def __init__(self, data_dir, tier="serving"):
        self.by_key = {}
        for name in ("train", "val", "test"):
            for line in open(Path(data_dir) / f"entries_{name}.jsonl", encoding="utf-8"):
                if line.strip():
                    e = json.loads(line)
                    if tier == "all" or e.get("serving"):
                        self.by_key[cards.norm(e["word"])] = e
        self.entries = list(self.by_key.values())
        self.sample_fn = cards.make_sampler(self.entries)

    def __len__(self):
        return len(self.entries)


class LobbyStore:
    """Remembers which words each lobby has seen. A word is never repeated while unused words remain;
    when the pool is nearly exhausted, the least-recently-used words are allowed back."""

    def __init__(self, ttl_s=6 * 3600, max_lobbies=5000):
        self.ttl, self.max = ttl_s, max_lobbies
        self.lobbies = OrderedDict()  # id -> (last_touch, OrderedDict(key -> None))
        self.lock = threading.Lock()

    def used(self, lobby_id):
        with self.lock:
            self._evict()
            item = self.lobbies.get(lobby_id)
            return list(item[1]) if item else []  # oldest first

    def add(self, lobby_id, keys):
        with self.lock:
            od = self.lobbies.get(lobby_id, (0, OrderedDict()))[1]
            for k in keys:
                od.pop(k, None)
                od[k] = None  # most recent at the end
            self.lobbies[lobby_id] = (time.time(), od)
            self.lobbies.move_to_end(lobby_id)

    def _evict(self):
        now = time.time()
        for k in [k for k, (t, _) in self.lobbies.items() if now - t > self.ttl]:
            del self.lobbies[k]
        while len(self.lobbies) > self.max:
            self.lobbies.popitem(last=False)


class Engine:
    def __init__(self, data_dir, ckpt=None, tokenizer=None, tier="serving", pool_size=0,
                 temperature=0.7, top_k=40, max_tries=3):
        self.verified = Verified(data_dir, tier)
        self.lobbies = LobbyStore()
        self.temperature, self.top_k, self.max_tries = temperature, top_k, max_tries
        self.model = None
        self.lock = threading.Lock()  # one generation at a time on the model
        self.pool, self.pool_size = [], pool_size
        self.stats = {"served": 0, "model_words": 0, "fill_words": 0, "generations": 0}
        if ckpt:
            import torch
            import generate as G
            torch.set_num_threads(max(1, torch.get_num_threads()))
            self.device = "cpu"
            self.net, self.tok = G.load(ckpt, tokenizer or str(HERE.parent / "train" / "tokenizer.json"), self.device)
            self._gen = G.generate_card_text
            self.model = True
        if self.model and pool_size:
            threading.Thread(target=self._refill_loop, daemon=True).start()

    # ---- the model's proposal -> verified entries only ----
    def _propose(self):
        with self.lock:
            text = self._gen(self.net, self.tok, self.device, self.temperature, self.top_k)
        self.stats["generations"] += 1
        found, seen = [], set()
        for w in WORD_RE.findall(text):
            e = self.verified.by_key.get(cards.norm(w))
            if e and cards.norm(w) not in seen:
                seen.add(cards.norm(w))
                found.append(e)
        return found

    def _refill_loop(self):
        while True:
            if len(self.pool) < self.pool_size:
                self.pool.append(self._propose())
            else:
                time.sleep(0.5)

    # ---- build one card ----
    def build_card(self, lobby_id=None):
        used_order = self.lobbies.used(lobby_id) if lobby_id else []
        used = set(used_order)
        chosen, keys = [], set()

        def take(entries):
            for e in entries:
                k = cards.norm(e["word"])
                if k not in used and k not in keys and len(chosen) < cards.CARD_SIZE:
                    keys.add(k)
                    chosen.append(e)

        if self.model:
            while self.pool and len(chosen) < cards.CARD_SIZE:
                take(self.pool.pop(0))
            tries = 0
            while len(chosen) < cards.CARD_SIZE and tries < self.max_tries:
                take(self._propose())
                tries += 1
        n_model = len(chosen)

        if len(chosen) < cards.CARD_SIZE:  # fill from the verified pool
            fresh = [e for e in self.verified.entries if cards.norm(e["word"]) not in used | keys]
            rng = random.Random()
            while len(chosen) < cards.CARD_SIZE and fresh:
                e = rng.choices(fresh, [cards.SERVING_WEIGHT if x.get("serving") else 1.0 for x in fresh])[0]
                fresh.remove(e)
                keys.add(cards.norm(e["word"]))
                chosen.append(e)
            if len(chosen) < cards.CARD_SIZE:  # pool exhausted: least-recently-used words may return
                for k in used_order:
                    if len(chosen) >= cards.CARD_SIZE:
                        break
                    if k not in keys and k in self.verified.by_key:
                        keys.add(k)
                        chosen.append(self.verified.by_key[k])
        random.shuffle(chosen)
        if lobby_id:
            self.lobbies.add(lobby_id, [cards.norm(e["word"]) for e in chosen])
        self.stats["served"] += 1
        self.stats["model_words"] += n_model
        self.stats["fill_words"] += len(chosen) - n_model
        card = {"words": [{"word": e["word"], "alias": e["alias"], "hint": e["hint"]} for e in chosen]}
        return card, n_model
