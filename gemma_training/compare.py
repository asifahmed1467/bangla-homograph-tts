"""Before/after table: did fine-tuning work?

Run after you have scored both models on the same test file:
    python evaluate.py --adapter unsloth/gemma-4-E2B-it --out predictions_before.csv   (original Gemma)
    python evaluate.py --out predictions.csv                                           (fine-tuned)
    python compare.py
Works the same with run_local.py results:  python compare.py --before local_before.csv --after local_after.csv
Writes comparison.md (a table you can show) and prints it.
"""
import argparse, csv, json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--before", default="predictions_before.csv")
ap.add_argument("--after", default="predictions.csv")
ap.add_argument("--out", default="comparison.md")
a = ap.parse_args()


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


before, after = load(a.before), load(a.after)
if [r["input"] for r in before] != [r["input"] for r in after]:
    raise SystemExit("The two files were made from different test sentences. Score both models on the same data/test.jsonl.")


def pct(rows, cond=lambda r: True):
    sel = [r for r in rows if cond(r)]
    c = sum(r["correct"] == "1" for r in sel)
    return c, len(sel), (100 * c / len(sel) if sel else 0.0)


from common import strip_tags

# "copy" = a model that changes nothing. About half the readings are spelled like the input,
# so copying already scores some points; a useful model must beat this.
copy = [dict(r, correct="1" if strip_tags(r["input"]) == r["expected"] else "0") for r in after]


def line(name, cond):
    if not pct(after, cond)[1]:
        return None
    c, b, a_ = pct(copy, cond), pct(before, cond), pct(after, cond)
    return (f"| {name} | {c[0]}/{c[1]} ({c[2]:.0f}%) | {b[0]}/{b[1]} ({b[2]:.0f}%) "
            f"| {a_[0]}/{a_[1]} ({a_[2]:.0f}%) | {a_[2] - b[2]:+.0f} points |")


out = ["# Did fine-tuning work?", "",
       f"Same {len(after)} test sentences, never seen during training. A sentence counts as correct only if every letter matches the dataset.", "",
       "| Test | Change nothing (baseline) | Original Gemma | Fine-tuned | Fine-tuned vs original |", "|---|---|---|---|---|"]
out += [l for l in (line("Homograph sentences fixed correctly", lambda r: r["kind"] == "homograph"),
                    line("Plain sentences left unchanged", lambda r: r["kind"] == "plain"),
                    line("All test sentences", lambda r: True)) if l]
out += ["", "*Change nothing* = just copying the input. The fine-tuned model has to beat this, too.", ""]

hb, ha = pct(before, lambda r: r["kind"] == "homograph")[2], pct(after, lambda r: r["kind"] == "homograph")[2]
pb, pa = pct(before, lambda r: r["kind"] == "plain"), pct(after, lambda r: r["kind"] == "plain")
hc = pct(copy, lambda r: r["kind"] == "homograph")[2]
verdict = "NO: the fine-tuned model is not better than changing nothing. Train more epochs or check the data." if ha <= hc else \
          "YES: the fine-tuned model is clearly better on homographs." if ha - hb >= 20 else \
          "PARTLY: some improvement; train longer or with more data." if ha - hb >= 5 else \
          "NO clear improvement yet: check the data, train more epochs, or use more words."
if pa[1] and pa[2] < 90:
    verdict += " Warning: it changes too many plain sentences; add more plain sentences."
out += [f"**Verdict: {verdict}**", ""]

words = sorted({r["word"] for r in after if r["word"]})
if words:
    out += ["## Per word", "", "| Word | Original | Fine-tuned |", "|---|---|---|"]
    for w in words:
        b, a_ = pct(before, lambda r: r["word"] == w), pct(after, lambda r: r["word"] == w)
        out.append(f"| {w} | {b[0]}/{b[1]} | {a_[0]}/{a_[1]} |")
    out.append("")

fixed = [(b, x) for b, x in zip(before, after) if b["correct"] == "0" and x["correct"] == "1"]
broken = [(b, x) for b, x in zip(before, after) if b["correct"] == "1" and x["correct"] == "0"]
still = [x for x in after if x["correct"] == "0"]
for title, pairs in (("Fixed by fine-tuning (examples)", fixed[:8]), ("Got worse after fine-tuning", broken[:8])):
    if pairs:
        out += [f"## {title}", ""]
        for b, x in pairs:
            out += [f"- input: {x['input']}", f"  - expected: {x['expected']}",
                    f"  - original: {b['predicted']}", f"  - fine-tuned: {x['predicted']}"]
        out.append("")
if still:
    out += [f"## Still wrong after fine-tuning ({len(still)}, first 8)", ""]
    for x in still[:8]:
        out += [f"- input: {x['input']}", f"  - expected: {x['expected']}", f"  - fine-tuned: {x['predicted']}"]
    out.append("")

text = "\n".join(out)
open(a.out, "w", encoding="utf-8").write(text)
print(text)
print(f"\nsaved: {a.out}")
