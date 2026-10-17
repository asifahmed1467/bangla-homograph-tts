"""Shared by app.py and test_examples.py: find homographs, put the tag in, call the model.

Flow (the same one the model was trained for, --tag_mode before):
  1. look for words from the homograph list in the sentence (whole words only)
  2. none found -> return the sentence unchanged, the model is NOT called
  3. found -> put <hg> directly in front of each one, ask the model, remove any <hg> it copied
"""
import csv, os, unicodedata
from pathlib import Path

import common
from common import strip_tags
from prepare_data import PUNCT, nfc

HERE = Path(__file__).resolve().parent
WORDS_CSV = HERE / "my_words_8.csv"      # the words the adapter was trained on
TAG = common.TAG or "<hg>"               # data/config.json (written by prepare_data.py) wins if it exists
# The prompt comes from common.py (+ data/config.json), exactly as in training, so use the same common.py
# the adapter was trained with. No config.json = no data/ folder: the tag stays <hg> and the prompt is common.py's default.


def load_words(path=WORDS_CSV):
    """Homograph column of the CSV -> set of NFC words. Also returns word -> bangla_phoneme of variant 1."""
    words, said = set(), {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            w = nfc(r["homograph"])
            words.add(w)
            if r.get("pronunciation_variant") == "1":
                said.setdefault(w, nfc(r["bangla_phoneme"]))
    return words, said


def tag_sentence(sentence, words, tag=TAG):
    """-> (tagged sentence, list of homographs found). A word counts only when it is a whole word:
    punctuation around it (। , ? ! quotes brackets) is ignored, parts of longer words are not matched."""
    found, out = [], []
    for token in sentence.split():
        core = token.strip(PUNCT)
        if core and unicodedata.normalize("NFC", core) in words:
            found.append(unicodedata.normalize("NFC", core))
            start = token.index(core)
            token = token[:start] + tag + token[start:]
        out.append(token)
    return " ".join(out), found


def load_model(adapter="gemma_cpu_lora", base="unsloth/gemma-3-270m-it"):
    """Gemma + LoRA adapter on CPU -> (rewrite, chat).
    rewrite(tagged_sentence): the fine-tuned model, answers with the respelled sentence.
    chat(text): the SAME loaded model with the adapter switched off = plain Gemma answering normally."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not os.path.isdir(adapter):
        raise SystemExit(f"adapter folder '{adapter}' not found. Train first (train_cpu.py) or pass --adapter <folder>.")
    bn = os.path.join(adapter, "base_model.txt")
    if os.path.exists(bn):
        base = open(bn).read().strip()
    torch.set_num_threads(os.cpu_count() or 4)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32), adapter)
    model.eval()

    @torch.no_grad()
    def generate(tagged):
        enc = tokenizer.apply_chat_template(common.messages(tagged), add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def chat(text):
        enc = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
        with model.disable_adapter():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return generate, chat


def load_answers(path=WORDS_CSV):
    """CSV rows as {normal-spelling sentence: sentence as the model should answer}, built the way
    prepare_data.py builds its training pairs. Used only by the stand-in model."""
    answers = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            tgt, word, said = nfc(r["sentence"]), nfc(r["homograph"]), nfc(r["bangla_phoneme"])
            src = " ".join(word.join(t.rsplit(said, 1)) if t.strip(PUNCT) == said else t for t in tgt.split())
            answers[src] = tgt
    return answers


def fake_chat(text):
    """STAND-IN for plain Gemma (used with --fake)."""
    return "[stand-in chat answer] " + text


def fake_model(said, answers=None):
    """STAND-IN, not Gemma. Answers from the CSV when it knows the exact sentence, otherwise respells the
    tagged word to its reading-1 spelling. Only for checking the lookup, tagging and page without the
    trained adapter. Use --fake to switch it on."""
    answers = answers if answers is not None else load_answers()

    def generate(tagged):
        plain = " ".join(tagged.replace(TAG, "").split())
        if plain in answers:
            return answers[plain]
        out = []
        for token in tagged.split():
            if TAG in token:
                core = token.replace(TAG, "").strip(PUNCT)
                token = token.replace(TAG, "").replace(core, said.get(core, core), 1)
            out.append(token)
        return " ".join(out)
    return generate


def fix_sentence(sentence, words, generate, chat=None, tag=TAG):
    """The whole pipeline. -> dict(input, found, tagged, called_model, mode, output).
    mode: "rewrite" (homograph found, fine-tuned model), "chat" (none found, plain Gemma answers),
    "unchanged" (none found and no chat model given: sentence returned as it is)."""
    sentence = nfc(sentence)
    tagged, found = tag_sentence(sentence, words, tag)
    if found:
        return {"input": sentence, "found": found, "tagged": tagged, "called_model": True, "mode": "rewrite",
                "output": strip_tags(generate(tagged))}
    if chat and sentence:
        return {"input": sentence, "found": [], "tagged": sentence, "called_model": True, "mode": "chat",
                "output": chat(sentence)}
    return {"input": sentence, "found": [], "tagged": sentence, "called_model": False, "mode": "unchanged",
            "output": sentence}


def get_models(adapter, fake, said):
    """-> (rewrite, chat)"""
    return (fake_model(said), fake_chat) if fake else load_model(adapter)
