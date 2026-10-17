"""Step 1: turn the homograph CSV into training files for Gemma.

Input  : tts_readiness/train_ready_rows.csv  (column sentence_homograph_form -> sentence_tts)
Output : data/train.jsonl, data/val.jsonl, data/test.jsonl
Each line: {"input": "<sentence as people type it>", "output": "<sentence with homograph respelled>", "kind": "homograph" | "plain", "word": ...}

Plain sentences (no homograph, output == input) teach the model to leave normal text alone.
They come from --plain_file (one sentence per line) or, with --wiki N, from Bangla Wikipedia.

Run:  python prepare_data.py --wiki 1500
      python prepare_data.py --words 12 --wiki 200            (quick test: small data, trains in minutes)
      python prepare_data.py --smoke                          (code check: tiny data, no internet needed)
      python prepare_data.py --smoke --only "কাশ,চট,শিখ,মত,হল"  (exactly these words + built-in plain sentences)
      python prepare_data.py --wiki 1500 --tag "<hg>" --tag_mode trigger   (tag anywhere in the sentence)
      python prepare_data.py --wiki 1500 --tag "<hg>" --tag_mode before    (tag right before the homograph: "যাচ্ছ <hg>কেন?")
      python prepare_data.py --wiki 1500 --tag "<hg>" --tag_mode mark      (tag wraps the homograph word)
The tag settings are saved in data/config.json, and train/evaluate/run_local read them from there.
"""
import argparse, csv, json, random, re, unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "tts_readiness" / "train_ready_rows.csv"
ALL_WORDS_CSV = HERE.parent / "homograph_call_context" / "bangla_1_to_160_combined_with_call_context.csv"

PUNCT = "।,;:!?\"'‘’“”()[]{}-–—.…"


def nfc(s):
    return unicodedata.normalize("NFC", s.strip())


def tokens(sentence):
    return [t.strip(PUNCT) for t in sentence.split()]


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def homograph_words():
    """Every homograph and respelled form we know of, so plain sentences never contain one."""
    words = set()
    for p in (DEFAULT_CSV, ALL_WORDS_CSV):
        if p.exists():
            for r in read_rows(p):
                for col in ("homograph", "bangla_phoneme"):
                    if r.get(col):
                        words.add(nfc(r[col]))
    return words


def homograph_position(sentence, target, homograph):
    """Index of the homograph word: the word that changes between input and target,
    or (when nothing changes, e.g. কর -> কর) the first word that equals the homograph."""
    words, tgt = sentence.split(), target.split()
    if len(words) == len(tgt):
        for i, (w, t) in enumerate(zip(words, tgt)):
            if w != t:
                return i
    for i, w in enumerate(words):
        if w.strip(PUNCT) == homograph:
            return i
    return None


def add_tag(sentence, homograph, tag, mode, rnd, target=""):
    """trigger: put the tag at a random word gap (start, middle or end).
    before: put the tag directly in front of the homograph word, e.g. "যাচ্ছ <hg>কেন?" (once).
    mark: wrap every copy of the homograph word as <hg>word</hg> (punctuation stays outside)."""
    words = sentence.split()
    if mode == "before":
        i = homograph_position(sentence, target or sentence, homograph)
        if i is not None:
            core = words[i].strip(PUNCT)
            start = words[i].index(core) if core else 0
            words[i] = words[i][:start] + tag + words[i][start:]
        return " ".join(words)
    if mode == "trigger":
        words.insert(rnd.randint(0, len(words)), tag)
        return " ".join(words)
    close = tag.replace("<", "</", 1) if tag.startswith("<") else tag
    out = []
    for w in words:
        core = w.strip(PUNCT)
        if homograph and core == homograph:
            start = w.index(core)
            w = w[:start] + tag + core + close + w[start + len(core):]
        out.append(w)
    return " ".join(out)


# Built-in plain sentences (no homograph) for the smoke test, so no internet download is needed.
BUILTIN_PLAIN = [
    "বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।",
    "আমি প্রতিদিন সকালে এক কাপ চা খাই।",
    "বৃষ্টির দিনে রাস্তায় অনেক পানি জমে যায়।",
    "আমার ছোট বোন স্কুলে নতুন বই পেয়েছে।",
    "গ্রামের মাঠে ছেলেরা ফুটবল খেলছে।",
    "নদীর তীরে একটি সুন্দর বাগান আছে।",
    "তিনি প্রতি শুক্রবার বাজারে যান।",
    "আজকের আবহাওয়া বেশ ঠান্ডা আর শুকনো।",
    "আমাদের বাড়ির সামনে একটি আমগাছ আছে।",
    "শিক্ষক ক্লাসে নতুন অধ্যায় শুরু করলেন।",
    "ডাক্তার রোগীকে নিয়মিত ওষুধ খেতে বললেন।",
    "ট্রেনটি সময়মতো স্টেশনে পৌঁছেছে।",
    "মা রান্নাঘরে ভাত আর ডাল রান্না করছেন।",
    "লাইব্রেরিতে অনেক পুরোনো বই রাখা আছে।",
    "শীতকালে খেজুরের রস খুব জনপ্রিয়।",
    "আমার বন্ধু আগামী মাসে চট্টগ্রাম যাবে।",
    "কৃষকেরা মাঠে ধান কাটছেন।",
    "সন্ধ্যায় আকাশে অনেক তারা দেখা যায়।",
    "দোকানে আজ তাজা মাছ এসেছে।",
    "ছাত্রছাত্রীরা মনোযোগ দিয়ে পরীক্ষা দিচ্ছে।",
    "বাগানে লাল আর হলুদ ফুল ফুটেছে।",
    "আমরা ছুটিতে সমুদ্রের ধারে বেড়াতে গিয়েছিলাম।",
    "নতুন সেতুটি দুই জেলাকে যুক্ত করেছে।",
    "দাদু প্রতিদিন বিকেলে হাঁটতে বের হন।",
]


def good_plain(sentence, banned):
    words = sentence.split()
    if not 5 <= len(words) <= 25:
        return False
    if re.search(r"[A-Za-z0-9০-৯=|<>{}\[\]]", sentence):
        return False
    bangla = sum("ঀ" <= ch <= "৿" for ch in sentence)
    if bangla < 0.7 * len(sentence.replace(" ", "")):
        return False
    return not any(t in banned for t in tokens(sentence))


def wiki_sentences(n, banned, seed, max_minutes=4):
    """Read Bangla Wikipedia articles one by one (streaming, no full download) until we have enough
    plain sentences, or until max_minutes have passed. Prints progress so you can see it is working."""
    import time
    from datasets import load_dataset  # pip install datasets

    print("reading Bangla Wikipedia (streaming)...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
    out, seen, start = [], set(), time.time()
    for i, art in enumerate(ds, 1):
        for s in re.split(r"(?<=[।?!])\s+", art["text"].replace("\n", " ")):
            s = nfc(s)
            if s not in seen and good_plain(s, banned):
                seen.add(s)
                out.append(s)
        if i % 200 == 0:
            print(f"  {i} articles read, {len(out)} plain sentences found", flush=True)
        if len(out) >= n * 3:
            break
        if time.time() - start > max_minutes * 60:
            print(f"  stopped after {max_minutes} minutes with {len(out)} sentences", flush=True)
            break
    random.Random(seed).shuffle(out)
    print(f"plain sentences from Wikipedia: {min(n, len(out))}", flush=True)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--input_col", default="sentence_homograph_form")
    ap.add_argument("--output_col", default="sentence_tts")
    ap.add_argument("--plain_file", help="text file, one plain Bangla sentence per line")
    ap.add_argument("--wiki", type=int, default=0, help="how many plain sentences to pull from Bangla Wikipedia")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--out_dir", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help='e.g. "<hg>". Avoid . , ! / @ (they appear in normal text)')
    ap.add_argument("--tag_mode", choices=["trigger", "before", "mark"], default="trigger")
    ap.add_argument("--words", type=int, default=0, help="quick test: use only this many homograph words (e.g. 12)")
    ap.add_argument("--only", default="", help='exact words to use, comma separated, e.g. "কাশ,চট,শিখ,মত,হল"')
    ap.add_argument("--smoke", action="store_true", help="code check: 3 words + built-in plain sentences, no download")
    a = ap.parse_args()
    if a.smoke:
        a.words, a.wiki = (a.words or (0 if (a.only or a.csv != str(DEFAULT_CSV)) else 3)), 0
    rnd = random.Random(a.seed)

    # 1. homograph pairs, grouped per word + reading so every reading appears in train AND test
    groups = defaultdict(dict)
    rows_in = read_rows(a.csv)
    derive = a.input_col not in rows_in[0]
    if derive:
        # plain dataset format (homograph, sentence, pronunciation_variant, bangla_phoneme):
        # target = sentence as written (with the pronunciation spelling),
        # input  = same sentence with the pronunciation spelling put back to the normal spelling
        print(f"{Path(a.csv).name}: no '{a.input_col}' column, so the input is made from 'sentence' + 'bangla_phoneme'")
    not_found = 0
    for r in rows_in:
        if derive:
            tgt, word, said = nfc(r["sentence"]), nfc(r["homograph"]), nfc(r["bangla_phoneme"])
            src = " ".join(word.join(t.rsplit(said, 1)) if t.strip(PUNCT) == said else t for t in tgt.split())
            if said != word and src == tgt:
                not_found += 1
                continue
        else:
            src, tgt = nfc(r[a.input_col]), nfc(r[a.output_col])
        if src and tgt:
            key = (nfc(r["homograph"]), r["pronunciation_variant"])
            groups[key][src] = {"input": src, "output": tgt, "kind": "homograph", "word": key[0], "variant": key[1]}
    if not_found:
        print(f"skipped {not_found} rows where the pronunciation spelling was not found in the sentence")

    # drop padded copies ("সত্যি বলতে, <same sentence>") so the same sentence can't be in train AND test
    dropped = 0
    for key, rows in groups.items():
        for src in list(rows):
            if ", " in src and src.split(", ", 1)[1] in rows:
                del rows[src]
                dropped += 1
    if dropped:
        print(f"removed {dropped} padded duplicate sentences (lead-in + an existing sentence)")

    if a.only:
        want = {nfc(w) for w in a.only.split(",") if w.strip()}
        missing = want - {k[0] for k in groups}
        if missing:
            print("WARNING: not in the data file (or not TRAIN_READY):", ", ".join(sorted(missing)))
        groups = {k: v for k, v in groups.items() if k[0] in want}
        print(f"using {len({k[0] for k in groups})} words: {', '.join(sorted({k[0] for k in groups}))}")
    elif a.words:
        # keep only words whose readings really differ in writing, so a quick test can show learning
        changing = sorted({k[0] for k, rows in groups.items() if any(r["input"] != r["output"] for r in rows.values())})
        keep = set(random.Random(a.seed).sample(changing, min(a.words, len(changing))))
        groups = {k: v for k, v in groups.items() if k[0] in keep}
        print(f"quick test: {len(keep)} words: {', '.join(sorted(keep))}")

    split = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        rows = list(groups[key].values())
        rnd.shuffle(rows)
        n_test = max(1, round(len(rows) * a.test_frac))
        n_val = max(1, round(len(rows) * a.val_frac)) if len(rows) >= 6 else 0
        split["test"] += rows[:n_test]
        split["val"] += rows[n_test:n_test + n_val]
        split["train"] += rows[n_test + n_val:]

    # 2. plain sentences: output == input
    banned = homograph_words()
    plain = []
    if a.plain_file:
        for line in open(a.plain_file, encoding="utf-8"):
            s = nfc(line)
            if s and good_plain(s, banned):
                plain.append(s)
    if a.smoke:
        plain += BUILTIN_PLAIN
    if a.wiki:
        plain += wiki_sentences(a.wiki, banned, a.seed)
    plain = list(dict.fromkeys(plain))
    rnd.shuffle(plain)
    if plain:
        n_test = round(len(plain) * a.test_frac)
        n_val = round(len(plain) * a.val_frac)
        for name, chunk in (("test", plain[:n_test]), ("val", plain[n_test:n_test + n_val]), ("train", plain[n_test + n_val:])):
            split[name] += [{"input": s, "output": s, "kind": "plain", "word": "", "variant": ""} for s in chunk]
    else:
        print("WARNING: no plain sentences. The model may start changing words it should leave alone. "
              "Use --wiki 1500 or --plain_file.")

    # 3. optional tag in the input (never in the output)
    if a.tag:
        if a.tag.strip(PUNCT + "/@#") == "":
            print(f"WARNING: tag {a.tag!r} is only punctuation. It also appears in normal sentences, use something like <hg>.")
        for rows in split.values():
            for r in rows:
                if a.tag_mode == "trigger" or r["kind"] == "homograph":
                    r["input"] = add_tag(r["input"], r["word"], a.tag, a.tag_mode, rnd, r["output"])
        # check: every homograph row has the tag exactly once, plain rows have none, outputs never have it
        if a.tag_mode == "before":
            bad, ok = [], 0
            for name, rows in split.items():
                for r in rows:
                    n = r["input"].count(a.tag)
                    want = 1 if r["kind"] == "homograph" else 0
                    if n != want or a.tag in r["output"]:
                        bad.append((name, r["input"]))
                    elif want:
                        ok += 1
            print(f"tag check: {ok} homograph rows tagged exactly once, {len(bad)} problems")
            for name, s in bad[:5]:
                print(f"  PROBLEM ({name}): {s}")
            for r in [r for r in split["train"] if r["kind"] == "homograph"][:3]:
                print(f"  example: {r['input']}  ->  {r['output']}")

    leak = {r["input"] for r in split["train"]} & {r["input"] for r in split["test"]}
    print(f"check: {len(leak)} test sentences also appear in training (should be 0)")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "tag_mode": a.tag_mode if a.tag else ""}, f, ensure_ascii=False)
    for name, rows in split.items():
        rnd.shuffle(rows)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["kind"] == "homograph" for r in rows)
        print(f"{name:5s}: {len(rows):5d} rows  ({h} homograph, {len(rows) - h} plain)")
    # the same data as one table you can open in Excel (UTF-8 with BOM so Bangla shows correctly)
    with open(out / "dataset_view.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "homograph_word", "input_with_hg_tag", "target", "tagged"])
        for name, rows in split.items():
            for r in rows:
                w.writerow([name, r["word"], r["input"], r["output"], "yes" if a.tag and a.tag in r["input"] else "no"])
    print(f"written to {out}  (open dataset_view.csv in Excel to see the inputs and targets)")


if __name__ == "__main__":
    main()
