"""Make the training files for the 2-tag model (sentences with up to two <hg> tags).

Mixes three kinds of rows, so the old behaviour keeps working:
  - two_tag_train.csv   new sentences with 2 tags (a few go to val), kind "two_tag"
  - train_dataset_hg_tag.csv   the old 1-tag and no-tag rows, kept in their old train/val/test split
  - two_tag_test.csv    new 2-tag sentences ONLY for testing, never trained on
Output: data/train.jsonl, data/val.jsonl, data/test.jsonl, data/config.json (tag <hg>, mode before), data/dataset_view.csv
Your old data/ folder is first copied to data_1tag_backup/ (once). The CSV files themselves are not changed.

Run:  python prepare_two_tag.py
"""
import argparse, csv, json, random, shutil, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = "<hg>"


def nfc(s):
    return unicodedata.normalize("NFC", s.strip())


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def new_row(r):
    return {"input": nfc(r["input_with_hg_tag"]), "output": nfc(r["target"]), "kind": "two_tag",
            "word": f"{r['word_1']}+{r['word_2']}", "variant": f"{r['reading_1']}+{r['reading_2']}"}


ap = argparse.ArgumentParser()
ap.add_argument("--new", default=str(HERE / "two_tag_train.csv"))
ap.add_argument("--test_csv", default=str(HERE / "two_tag_test.csv"))
ap.add_argument("--old", default=str(HERE / "train_dataset_hg_tag.csv"))
ap.add_argument("--val", type=int, default=16, help="how many new 2-tag rows go to val instead of train")
ap.add_argument("--out_dir", default=str(HERE / "data"))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rnd = random.Random(a.seed)

out = Path(a.out_dir)
backup = out.parent / "data_1tag_backup"
if out.exists() and not backup.exists():
    shutil.copytree(out, backup)
    print(f"old data folder copied to {backup}")

split = {"train": [], "val": [], "test": []}
for r in read(a.old):
    tagged = TAG in r["input_with_hg_tag"]
    split[r["split"]].append({"input": nfc(r["input_with_hg_tag"]), "output": nfc(r["target"]),
                              "kind": "homograph" if tagged else "plain",
                              "word": r["homograph_word"] if tagged else "", "variant": ""})
new = [new_row(r) for r in read(a.new)]
rnd.shuffle(new)
split["val"] += new[:a.val]
split["train"] += new[a.val:]
split["test"] += [new_row(r) for r in read(a.test_csv)]

# checks: 2-tag rows have exactly 2 tags, no target has a tag, no test sentence is trained on
bad = [r["input"] for rows in split.values() for r in rows
       if TAG in r["output"] or (r["kind"] == "two_tag" and r["input"].count(TAG) != 2)]
leak = {r["input"] for r in split["train"]} & {r["input"] for r in split["test"]}
print(f"check: {len(bad)} rows with a wrong tag count (should be 0), "
      f"{len(leak)} test sentences also in training (should be 0)")
for s in bad[:5]:
    print("  PROBLEM:", s)

out.mkdir(parents=True, exist_ok=True)
with open(out / "config.json", "w", encoding="utf-8") as f:
    json.dump({"tag": TAG, "tag_mode": "before"}, f, ensure_ascii=False)
for name, rows in split.items():
    rnd.shuffle(rows)
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = {k: sum(r["kind"] == k for r in rows) for k in ("two_tag", "homograph", "plain")}
    print(f"{name:5s}: {len(rows):4d} rows  ({n['two_tag']} two-tag, {n['homograph']} one-tag, {n['plain']} no-tag)")
with open(out / "dataset_view.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "kind", "word", "input_with_hg_tag", "target"])
    for name, rows in split.items():
        for r in rows:
            w.writerow([name, r["kind"], r["word"], r["input"], r["output"]])
print(f"written to {out}  (open dataset_view.csv in Excel to see everything)")
