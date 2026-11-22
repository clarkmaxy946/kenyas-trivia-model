#!/usr/bin/env python3
"""
Latency under concurrent load (spec evaluation part (a)).

  python eval/loadtest.py --url http://127.0.0.1:8000 --users 1 5 10 --requests 10

For each concurrency level, N simulated lobbies each send --requests card requests at once.
Prints median / 95th percentile / max latency, requests per second, and failures, and checks every
reply against the Card Contract. Paste the table into the evaluation chapter.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import cards  # noqa: E402


def one(url, lobby):
    body = json.dumps({"lobby_id": lobby}).encode()
    req = urllib.request.Request(url + "/generate_card", body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read().decode()
        src = r.headers.get("X-Card-Source", "")
    dt = time.time() - t
    obj, probs = cards.check_card(data)
    return dt, obj is not None, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--requests", type=int, default=10, help="requests per user")
    a = ap.parse_args()
    print(f"{'users':>5} {'requests':>8} {'median s':>9} {'p95 s':>7} {'max s':>7} {'req/s':>6} {'failed':>6} {'invalid':>7} {'model words':>11}")
    for n in a.users:
        lat, bad, fail, model_words = [], 0, 0, 0
        lock = threading.Lock()

        def user(i):
            nonlocal bad, fail, model_words
            for _ in range(a.requests):
                try:
                    dt, ok, src = one(a.url, f"load-{n}-{i}")
                except Exception:
                    with lock:
                        fail += 1
                    continue
                with lock:
                    lat.append(dt)
                    bad += (not ok)
                    model_words += int(src.split("model=")[1].split(";")[0]) if "model=" in src else 0

        t0 = time.time()
        th = [threading.Thread(target=user, args=(i,)) for i in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.time() - t0
        if lat:
            lat.sort()
            p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
            share = 100 * model_words / (10 * len(lat))
            print(f"{n:>5} {len(lat):>8} {statistics.median(lat):>9.2f} {p95:>7.2f} {lat[-1]:>7.2f} {len(lat) / wall:>6.1f} {fail:>6} {bad:>7} {share:>10.0f}%")
        else:
            print(f"{n:>5} all requests failed")


if __name__ == "__main__":
    main()
