"""Score a sahajBERT model on the held-out two-tag sentences (never used in training).

Run:  python evaluate_2tag.py --model model --out before_2tag.csv        (old model, for comparison)
      python evaluate_2tag.py --model model_2tag --out after_2tag.csv    (new model)
"""
import argparse, csv, sys, time
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemma_training"))
from two_tags import one_tag_copies, put_respellings
from src.common import read_csv
from predict_sentence import Predictor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("HG_SAHAJBERT_MODEL") or str(HERE / "model_2tag"))
ap.add_argument("--test", default=str(HERE.parent / "gemma_training" / "two_tag_test.csv"))
ap.add_argument("--lexicon", default=str(HERE / "data" / "lexicon.json"))
ap.add_argument("--out", default="predictions_2tag.csv")
a = ap.parse_args()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

rows = read_csv(a.test)
p = Predictor(a.model, a.lexicon)
copies = [c for r in rows for c in one_tag_copies(r["input_with_hg_tag"])]
t = time.time()
results = p.predict(copies)                      # the whole test set in one batch call
secs = time.time() - t

per_word = defaultdict(lambda: [0, 0])
full = 0
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["input", "expected", "predicted", "correct"])
    for i, r in enumerate(rows):
        pair = results[2 * i:2 * i + 2]
        for res, word, reading in zip(pair, (r["word_1"], r["word_2"]), (r["reading_1"], r["reading_2"])):
            per_word[word][0] += res["pronunciation_variant"] == int(reading)
            per_word[word][1] += 1
        pred = put_respellings(r["input_with_hg_tag"], [res["bangla_phoneme"] for res in pair])
        ok = pred == r["target"]
        full += ok
        w.writerow([r["input_with_hg_tag"], r["target"], pred, int(ok)])

words_ok = sum(c for c, _ in per_word.values())
words_n = sum(n for _, n in per_word.values())
print(f"two-tag sentences fully right: {full}/{len(rows)} = {full / len(rows):.1%}")
print(f"tagged words right: {words_ok}/{words_n} = {words_ok / words_n:.1%}")
print("per word:", ", ".join(f"{wd} {c}/{n}" for wd, (c, n) in sorted(per_word.items())))
print(f"time: {secs:.2f} s for {len(rows)} sentences = {1000 * secs / len(rows):.0f} ms per sentence")
print("all predictions:", a.out)
