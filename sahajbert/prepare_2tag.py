"""Add the two-tag sentences to sahajBERT's training data.

sahajBERT picks the reading of ONE tagged word, so each two-tag sentence becomes 2 training rows:
one with only the first word tagged, one with only the second (the other word stays in normal spelling,
exactly as it will be typed). The old rows from prepare.py are kept, so one-tag sentences keep working.

Run (after python prepare.py):  python prepare_2tag.py
In : data/processed.csv (from prepare.py), ../gemma_training/two_tag_train.csv
Out: data/processed_2tag.csv   (then: python train.py --data data/processed_2tag.csv --epochs 7 --batch-size 8 --out model_2tag_v3)
"""
import argparse, csv, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemma_training"))
from two_tags import one_tag_copies, tagged_words
from src.common import read_csv

ap = argparse.ArgumentParser()
ap.add_argument("--old", default=str(HERE / "data" / "processed.csv"))
ap.add_argument("--new", default=str(HERE.parent / "gemma_training" / "two_tag_train.csv"))
ap.add_argument("--lexicon", default=str(HERE / "data" / "lexicon.json"))
ap.add_argument("--out", default=str(HERE / "data" / "processed_2tag.csv"))
ap.add_argument("--val", type=int, default=16, help="how many two-tag sentences (2 rows each) go to validation")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

if not Path(a.old).exists():
    sys.exit(f"{a.old} not found: run  python prepare.py  first")
lex = json.loads(Path(a.lexicon).read_text(encoding="utf-8"))
old = read_csv(a.old)
new = read_csv(a.new)
random.Random(a.seed).shuffle(new)

rows, problems = [], 0
for i, r in enumerate(new):
    split = "val" if i < a.val else "train"
    sentence = r["input_with_hg_tag"].strip()
    copies = one_tag_copies(sentence)
    words = tagged_words(sentence)
    readings = [r["reading_1"], r["reading_2"]]
    if len(copies) != 2 or words != [r["word_1"], r["word_2"]]:
        problems += 1
        print("PROBLEM:", sentence)
        continue
    for copy, word, reading in zip(copies, words, readings):
        m = lex[word][reading]
        rows.append({"split": split, "homograph": word, "tagged_sentence": copy, "pronunciation_variant": int(reading),
                     "phoneme": m["phoneme"], "bangla_phoneme": m["bangla_phoneme"]})

fields = ["split", "homograph", "tagged_sentence", "pronunciation_variant", "phoneme", "bangla_phoneme"]
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows([{k: r[k] for k in fields} for r in old] + rows)
count = lambda rs, s: sum(r["split"] == s for r in rs)
print(f"old rows: train {count(old, 'train')}, val {count(old, 'val')}")
print(f"new rows from two-tag sentences: train {count(rows, 'train')}, val {count(rows, 'val')}  ({problems} problems)")
print("saved:", a.out)
