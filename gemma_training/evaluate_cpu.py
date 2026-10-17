"""Score a small Gemma on the test sentences, on CPU. Pairs with train_cpu.py.

Run:  python evaluate_cpu.py --out predictions_before.csv               (original model, no training)
      python evaluate_cpu.py --adapter gemma_cpu_lora --out predictions.csv   (fine-tuned)
      python compare.py                                                   (before/after table)
      python evaluate_cpu.py --adapter gemma_cpu_lora --sentence "..."    (one sentence)
"""
import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import messages, load_jsonl, score_rows

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/gemma-3-270m-it", help="base model (ignored when --adapter has base_model.txt)")
ap.add_argument("--adapter", help="folder made by train_cpu.py; leave out to test the original model")
ap.add_argument("--test", default="data/test.jsonl")
ap.add_argument("--out", default="predictions.csv")
ap.add_argument("--sentence")
ap.add_argument("--limit", type=int, default=0, help="code check: score only the first N test sentences")
a = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 4)
base = a.model
if a.adapter and os.path.exists(os.path.join(a.adapter, "base_model.txt")):
    base = open(os.path.join(a.adapter, "base_model.txt")).read().strip()
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()


@torch.no_grad()
def rewrite(sentence):
    enc = tokenizer.apply_chat_template(messages(sentence), add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    out = model.generate(**enc, max_new_tokens=128, do_sample=False)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if a.sentence:
    print(rewrite(a.sentence))
else:
    rows = load_jsonl(a.test)
    score_rows(rows[:a.limit] if a.limit else rows, rewrite, a.out, label=a.adapter or base)
