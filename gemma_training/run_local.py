"""Use the trained model on your own PC (no Python libraries needed).

1. Install llama.cpp:   winget install llama.cpp      (Windows)   |   brew install llama.cpp   (Mac)
2. Start the model in one terminal (GTX 1050 Ti example):
       llama-server -m gemma_homograph_gguf/<file>.gguf -ngl 99 -c 1024
   -ngl 99 = put all layers on the GPU. "out of memory"? use -ngl 20 (part GPU, part CPU) or -ngl 0 (CPU only, slower).
3. In a second terminal:
       python run_local.py "আসর নামাজের পর আমরা বাজারে যাব।"
       python run_local.py --file sentences.txt        (one sentence per line, writes sentences_fixed.txt)
       python run_local.py --test data/test.jsonl --out local_after.csv
                                                       (score the local model on the held-out test sentences)
"""
import argparse, json, urllib.request
from common import INSTRUCTION, strip_tags, load_jsonl, score_rows

URL = "http://127.0.0.1:8080/v1/chat/completions"


def rewrite(sentence):
    body = {
        "messages": [{"role": "user", "content": INSTRUCTION + sentence}],
        "temperature": 0,            # always pick the most likely answer, no randomness
        "max_tokens": 128,
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return strip_tags(json.load(r)["choices"][0]["message"]["content"].strip())


ap = argparse.ArgumentParser()
ap.add_argument("sentence", nargs="?")
ap.add_argument("--file")
ap.add_argument("--test", help="data/test.jsonl: score the model that llama-server is running")
ap.add_argument("--out", default="local_predictions.csv")
a = ap.parse_args()
if a.test:
    score_rows(load_jsonl(a.test), rewrite, a.out, label=a.out.rsplit(".", 1)[0])
elif a.file:
    out = a.file.rsplit(".", 1)[0] + "_fixed.txt"
    with open(a.file, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if line.strip():
                g.write(rewrite(line.strip()) + "\n")
    print("written", out)
elif a.sentence:
    print(rewrite(a.sentence))
else:
    ap.print_help()
