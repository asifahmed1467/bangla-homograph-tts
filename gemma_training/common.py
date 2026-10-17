"""Shared settings so training and testing use exactly the same prompt."""

BASE_MODEL = "unsloth/gemma-4-E2B-it"   # bigger GPU (16 GB+)? try "unsloth/gemma-4-E4B-it"
ADAPTER_DIR = "gemma_homograph_lora"    # where the trained LoRA adapter is saved
MAX_SEQ_LEN = 256                       # your sentences are short; this keeps memory low

INSTRUCTION = (
    "নিচের বাংলা বাক্যে কোনো হোমোগ্রাফ শব্দ থাকলে প্রসঙ্গ অনুযায়ী সেটিকে উচ্চারণ-বানানে লেখো। "
    "অন্য কোনো শব্দ বদলাবে না। হোমোগ্রাফ না থাকলে বাক্যটি হুবহু ফেরত দাও।\n\n"
)
# English: "If the Bangla sentence below has a homograph, rewrite it in its pronunciation
# spelling according to context. Change nothing else. If there is no homograph, return the
# sentence exactly as it is."


# Tag settings written by prepare_data.py (data/config.json). No file = no tag.
import json as _json
from pathlib import Path as _Path

_cfg_path = _Path(__file__).resolve().parent / "data" / "config.json"
_cfg = _json.loads(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}
TAG = _cfg.get("tag", "")
TAG_MODE = _cfg.get("tag_mode", "")

if TAG_MODE in ("trigger", "before"):
    # the tag itself is the signal (before: it sits right in front of the word to respell), so the long instruction is not needed
    INSTRUCTION = ""
elif TAG_MODE == "mark":
    _close = TAG.replace("<", "</", 1) if TAG.startswith("<") else TAG
    INSTRUCTION = (
        f"নিচের বাক্যে {TAG}...{_close} দিয়ে চিহ্নিত শব্দটি প্রসঙ্গ অনুযায়ী উচ্চারণ-বানানে লেখো। "
        "চিহ্ন দুটো বাদ দিয়ে পুরো বাক্যটি ফেরত দাও। অন্য কোনো শব্দ বদলাবে না।\n\n"
    )


def strip_tags(text):
    """Safety net: remove any tag the model copied into its answer."""
    if not TAG:
        return text
    close = TAG.replace("<", "</", 1) if TAG.startswith("<") else TAG
    return " ".join(text.replace(close, "").replace(TAG, " ").split())


def messages(sentence, answer=None):
    msgs = [{"role": "user", "content": INSTRUCTION + sentence}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def load_jsonl(path):
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_rows(rows, rewrite, out_csv, label):
    """Run `rewrite` on every test row, write a CSV of predictions and a JSON summary, print the scores."""
    import csv, json
    from collections import defaultdict
    score = defaultdict(lambda: [0, 0])          # "homograph" / "plain" / "word:<w>" -> [correct, total]
    leaked = 0
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "word", "variant", "input", "expected", "predicted", "correct"])
        for i, r in enumerate(rows, 1):
            pred = rewrite(r["input"])
            ok = pred == r["output"]             # strict: every letter must match, a copied tag is a mistake
            leaked += bool(TAG and TAG in pred)
            for k in (r["kind"], "word:" + r["word"]) if r["word"] else (r["kind"],):
                score[k][0] += ok
                score[k][1] += 1
            w.writerow([r["kind"], r["word"], r["variant"], r["input"], r["output"], pred, int(ok)])
            if i % 25 == 0:
                print(f"{i}/{len(rows)}")
    summary = {"label": label, "csv": out_csv, "tag_leaked": leaked, **{k: v for k, v in score.items()}}
    with open(out_csv.rsplit(".", 1)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for k, name in (("homograph", "homograph sentences fixed correctly"), ("plain", "plain sentences left unchanged")):
        if score[k][1]:
            c, n = score[k]
            print(f"{name}: {c}/{n} = {c / n:.1%}")
    if leaked:
        print(f"answers that still contain the tag {TAG}: {leaked}")
    weak = sorted((c / n, k[5:]) for k, (c, n) in score.items() if k.startswith("word:") and c < n)
    print("words with mistakes:", ", ".join(f"{wd} ({acc:.0%})" for acc, wd in weak[:20]) or "none")
    print("all predictions:", out_csv)
    return summary
