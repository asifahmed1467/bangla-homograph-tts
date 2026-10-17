"""Step 5 (optional): pack base model + your LoRA into one GGUF file you can run on your own PC.

Run on Colab after train.py:  python export_gguf.py
Output: gemma_homograph_gguf/*.gguf  (about 3 GB at q4_k_m)
"""
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from common import ADAPTER_DIR, MAX_SEQ_LEN

model, tokenizer = FastModel.from_pretrained(model_name=ADAPTER_DIR, max_seq_length=MAX_SEQ_LEN, load_in_4bit=False)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")   # same template as training
# q4_k_m = 4-bit, about 3.1 GB. Use "q3_k_m" (about 2.5 GB) if it does not fit a 4 GB GPU, "q8_0" for best quality.
model.save_pretrained_gguf("gemma_homograph_gguf", tokenizer, quantization_method="q4_k_m")
print("done: see gemma_homograph_gguf/")
