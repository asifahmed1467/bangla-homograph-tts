"""Step 3: test a model on sentences it never saw.

Run:  python evaluate.py                                   -> fine-tuned model, writes predictions.csv
      python evaluate.py --adapter unsloth/gemma-4-E2B-it --out predictions_before.csv
                                                           -> original Gemma (no training), for comparison
      python evaluate.py --sentence "..."                  -> rewrite one sentence
Then: python compare.py   -> before/after table
"""
import argparse
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from common import ADAPTER_DIR, MAX_SEQ_LEN, messages, load_jsonl, score_rows

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=ADAPTER_DIR, help="adapter folder, or a base model name to test without training")
ap.add_argument("--test", default="data/test.jsonl")
ap.add_argument("--out", default="predictions.csv")
ap.add_argument("--sentence")
a = ap.parse_args()

model, tokenizer = FastModel.from_pretrained(model_name=a.adapter, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
FastModel.for_inference(model)


def rewrite(sentence):
    prompt = tokenizer.apply_chat_template(messages(sentence), tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text=[prompt.removeprefix("<bos>")], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=128, do_sample=False)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if a.sentence:
    print(rewrite(a.sentence))
else:
    score_rows(load_jsonl(a.test), rewrite, a.out, label=a.adapter)
