"""Code check: does the whole fine-tuning pipeline run on this PC? (CPU, no GPU, a few minutes)

Run:  python smoke_test.py
It runs every step on tiny data and prints PASS / FAIL for each check at the end.
This proves the CODE works. It does not prove the model is good: 10 training steps are far too few for that.

Uses unsloth/gemma-3-270m-it (Gemma 3 270M, no Hugging Face login needed, about 0.6 GB download).
Other model:  python smoke_test.py --model google/gemma-3-270m-it   (needs licence + `hf auth login`)
"""
import argparse, csv, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/gemma-3-270m-it")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--limit", type=int, default=8, help="test sentences scored before and after")
a = ap.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
results = []


def run(title, args):
    print(f"\n=== {title} ===\n> python {' '.join(args)}", flush=True)
    t = time.time()
    ok = subprocess.run([PY] + args).returncode == 0
    print(f"--- {'done' if ok else 'FAILED'} in {time.time() - t:.0f} s", flush=True)
    results.append((title, ok))
    return ok


def check(title, ok, detail=""):
    results.append((title + (f" ({detail})" if detail else ""), bool(ok)))


steps_ok = (
    run("1. Prepare tiny dataset", ["prepare_data.py", "--smoke"])
    and run("2. Score the ORIGINAL model", ["evaluate_cpu.py", "--model", a.model, "--limit", str(a.limit),
                                             "--out", "smoke_before.csv"])
    and run("3. Fine-tune (LoRA) for a few steps", ["train_cpu.py", "--model", a.model, "--max_steps", str(a.steps),
                                                    "--out", "smoke_lora"])
    and run("4. Reload the saved adapter and score it", ["evaluate_cpu.py", "--adapter", "smoke_lora",
                                                          "--limit", str(a.limit), "--out", "smoke_after.csv"])
    and run("5. Before/after table", ["compare.py", "--before", "smoke_before.csv", "--after", "smoke_after.csv",
                                      "--out", "smoke_comparison.md"])
)

if steps_ok:
    check("data files written", all(os.path.exists(f"data/{n}.jsonl") for n in ("train", "val", "test")))
    check("adapter files saved", os.path.exists("smoke_lora/adapter_config.json")
          and any(f.startswith("adapter_model") for f in os.listdir("smoke_lora")))
    losses = json.load(open("smoke_lora/train_log.json"))["losses"] if os.path.exists("smoke_lora/train_log.json") else []
    if len(losses) >= 4:
        first, last = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
        check("loss went down", last < first, f"{first:.2f} -> {last:.2f}")
    with open("smoke_after.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    check("predictions file written", len(rows) > 0, f"{len(rows)} sentences")

print("\n================ SMOKE TEST RESULT ================")
for title, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {title}")
allok = all(ok for _, ok in results)
print("====================================================")
print("ALL CHECKS PASSED: the fine-tuning code works on this PC." if allok else
      "Something failed. Copy the red error text above the first FAIL and send it.")
print("Scores after 10 steps are not meaningful; they only show the pipeline runs end to end.")
sys.exit(0 if allok else 1)
