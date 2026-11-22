#!/usr/bin/env python3
"""
Phase 3, step 2 (laptop): train the model's own tokenizer (byte-level BPE, 8,192 tokens).

Trained only on our own text: the pretraining articles + sample cards from the TRAIN entries.
Small vocabulary on purpose: with ~4M words of text, a big vocabulary would leave most tokens
rarely seen. Byte-level means any character (Kĩkũyũ names, accents) can always be encoded.

Output: train/tokenizer.json   (copy it to Kaggle together with train/ and data/corpus/)
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import cards  # noqa: E402

C = ROOT / "data" / "corpus"
VOCAB = 8192


def main():
    entries = [json.loads(l) for l in open(C / "entries_train.jsonl", encoding="utf-8") if l.strip()]
    sample = cards.make_sampler(entries)
    rng = cards.rng_from(7)
    card_file = C / "_tokenizer_cards.txt"
    with open(card_file, "w", encoding="utf-8") as f:
        for _ in range(1500):
            f.write(cards.card_to_text(sample(rng)) + "\n")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, min_frequency=2, show_progress=False,
        special_tokens=[cards.PAD, cards.BOS, cards.EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(C / "pretrain_train.txt"), str(card_file)], trainer)
    tok.save(str(ROOT / "train" / "tokenizer.json"))
    card_file.unlink()

    # report: how long is a card in tokens? (decides the model's context length)
    lens = []
    rng = cards.rng_from(99)
    for _ in range(300):
        lens.append(len(tok.encode(cards.card_to_text(sample(rng))).ids) + 2)
    lens.sort()
    n = len(tok.encode(open(C / "pretrain_val.txt", encoding="utf-8").read()).ids)
    print(f"vocab {tok.get_vocab_size()} | card length in tokens: median {lens[150]}, p95 {lens[285]}, max {lens[-1]}")
    print(f"pretrain_val.txt = {n / 1e6:.2f}M tokens  (pretrain_train is ~{n / 0.0196 / 1e6:.1f}M tokens)")
    s = 'Kenya\'s Trivia: Kĩkũyũ Nyong\'o Mũgumo "Uhuru"'
    ids = tok.encode(s).ids
    print("round trip ok:", tok.decode(ids) == s)


if __name__ == "__main__":
    main()
