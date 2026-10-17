"""Fine-tune a SMALL Gemma on your own PC, no GPU needed (CPU only). For the quick test.

Gemma 4 E2B is too big for CPU training (it needs more than 16 GB RAM and would take days),
so this uses Gemma 3 270M (default) or Gemma 3 1B. Smaller model = weaker Bangla,
but it is enough to check that fine-tuning works.

Run (after prepare_data.py --words 12 --wiki 200):
    python train_cpu.py                                   (Gemma 3 270M, about 1 GB RAM for the model)
    python train_cpu.py --max_steps 10                    (code check only: 10 steps, about a minute)
    python train_cpu.py --model google/gemma-3-1b-it      (better, about 4x slower)
"""
import argparse, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from common import MAX_SEQ_LEN, messages, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/gemma-3-270m-it")   # same weights as google/gemma-3-270m-it, no login needed
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", default="gemma_cpu_lora")
ap.add_argument("--max_steps", type=int, default=-1, help="code check: stop after this many steps (e.g. 10)")
a = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 4)
tokenizer = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)   # CPU: plain float32

# LoRA: freeze the model, add small trainable layers
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()


def encode(row):
    """Token ids for prompt + answer. labels = -100 on the prompt, so the model only learns the answer."""
    prompt = tokenizer.apply_chat_template(messages(row["input"]), tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages(row["input"], row["output"]), tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels[:len(ids)]}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
    }


train_ds = [encode(r) for r in load_jsonl("data/train.jsonl")]
val_ds = [encode(r) for r in load_jsonl("data/val.jsonl")]
steps = a.max_steps if a.max_steps > 0 else int(len(train_ds) / a.batch * a.epochs)
print(f"{len(train_ds)} training examples, {a.epochs} epochs, about {steps} steps on {torch.get_num_threads()} CPU threads")

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
    args=TrainingArguments(
        output_dir="checkpoints_cpu",
        use_cpu=True,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=2e-4,
        warmup_steps=5,
        lr_scheduler_type="linear",
        logging_steps=1 if a.max_steps > 0 else 5,
        eval_strategy="no" if a.max_steps > 0 else "epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    ),
)
t = time.time()
trainer.train()
print(f"training took {(time.time() - t) / 60:.1f} minutes")
model.save_pretrained(a.out)
tokenizer.save_pretrained(a.out)
with open(os.path.join(a.out, "base_model.txt"), "w") as f:
    f.write(a.model)
import json
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
with open(os.path.join(a.out, "train_log.json"), "w") as f:
    json.dump({"losses": losses}, f)
if losses:
    print(f"loss: first {losses[0]:.3f} -> last {losses[-1]:.3f}")
print(f"saved adapter to {a.out}/")
