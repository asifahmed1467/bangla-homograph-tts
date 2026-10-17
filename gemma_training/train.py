"""Step 2: fine-tune Gemma 4 with QLoRA (4-bit base model + small trainable LoRA layers).

Run:  python train.py              (full data)
      python train.py --epochs 2     (quick test)
Needs an NVIDIA GPU with 8 GB+ VRAM (Unsloth: "Gemma 4 E2B trains on 8GB VRAM").
A 4 GB card such as a GTX 1050 Ti is too small: use colab_train.ipynb on free Google Colab instead.
"""
from unsloth import FastModel, is_bf16_supported  # import unsloth first, it patches the others
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from common import BASE_MODEL, ADAPTER_DIR, MAX_SEQ_LEN, messages, load_jsonl
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=float, default=3, help="quick test: 2")
args = ap.parse_args()

# 1. Load the base model in 4-bit so it fits on a consumer GPU
model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

# 2. Add LoRA adapters: only these small layers learn, the original weights stay frozen
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # text only
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    random_state=42,
)

# 3. Turn each {"input", "output"} pair into Gemma's chat format
def to_text(rows):
    texts = []
    for r in rows:
        t = tokenizer.apply_chat_template(messages(r["input"], r["output"]), tokenize=False)
        texts.append(t.removeprefix("<bos>"))   # the tokenizer adds <bos> itself
    return Dataset.from_dict({"text": texts})

train_ds = to_text(load_jsonl("data/train.jsonl"))
val_ds = to_text(load_jsonl("data/val.jsonl"))
print("example training text:\n", train_ds[0]["text"])

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        per_device_train_batch_size=8,     # out of memory? use 4 or 2 and raise the next line
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,      # small data: 2-4 epochs; watch eval_loss
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=is_bf16_supported(),          # newer GPUs (RTX 30xx+)
        fp16=not is_bf16_supported(),      # older GPUs and the free Colab T4
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="checkpoints",
        seed=42,
        report_to="none",
    ),
)
# learn only the answer, not the instruction
trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
trainer.train()

# 5. Save only the small adapter (tens of MB), not the whole model
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"saved adapter to {ADAPTER_DIR}/")
