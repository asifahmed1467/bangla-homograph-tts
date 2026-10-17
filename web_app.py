r"""One web page for all three models (the Docker image runs this file).

Pick the model in the dropdown, type one sentence or many (one per line), press Run.
  - Gemma 3 270M, plain chat:   the adapter is switched off, Gemma answers like a normal chat.
  - Gemma + homograph adapter:  a line WITH <hg> is respelled by the adapter (tag removed in the output);
                                a line WITHOUT <hg> goes to plain Gemma, which does its normal work.
  - sahajBERT classifier:       all lines WITH <hg> go to the model in ONE call (list in, list out);
                                lines without <hg> come back unchanged (sahajBERT only does homographs).

Run without Docker:  python web_app.py           (models are read from the models/ folder next to this file)
                     set HG_FAKE=1 then python web_app.py   (stand-in answers, only to look at the page)
Then open http://127.0.0.1:7860

Settings (environment variables, docker-compose.yml sets them):
  HG_MODELS_DIR       folder holding gemma_cpu_lora/ and sahajbert/      (default: models/ next to this file)
  HG_GEMMA_ADAPTER    the Gemma LoRA folder                               (default: <models>/gemma_cpu_lora)
  HG_SAHAJBERT_MODEL  the trained sahajBERT folder                        (default: <models>/sahajbert)
  HG_GEMMA_BASE       base model when the adapter has no base_model.txt   (default: unsloth/gemma-3-270m-it)
  HG_TAG_MODE         how the adapter was trained: trigger, before or none (default: before, the retrained adapter;
                      used only when gemma_training/data/config.json does not exist)
  HG_SHARE=1          also print a temporary public gradio.live link
  HG_FAKE=1           stand-in models, no download, only to check the page
"""
import contextlib, html, os, re, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEMMA_DIR = ROOT / "gemma_training"
SAHAJ_DIR = ROOT / "sahajbert"
MODELS_DIR = Path(os.environ.get("HG_MODELS_DIR") or ROOT / "models")
GEMMA_ADAPTER = Path(os.environ.get("HG_GEMMA_ADAPTER") or MODELS_DIR / "gemma_cpu_lora")
SAHAJ_MODEL = Path(os.environ.get("HG_SAHAJBERT_MODEL") or MODELS_DIR / "sahajbert")
GEMMA_BASE = os.environ.get("HG_GEMMA_BASE", "unsloth/gemma-3-270m-it")
TAG_MODE = os.environ.get("HG_TAG_MODE", "before").strip().lower()
FAKE = os.environ.get("HG_FAKE") == "1"
SHARE = os.environ.get("HG_SHARE") == "1"

# sahajBERT reads its settings through src/config.py, where HG_<KEY> overrides a key
os.environ["HG_MODEL_DIR"] = str(SAHAJ_MODEL)
sys.path[:0] = [str(GEMMA_DIR), str(SAHAJ_DIR)]

import gradio as gr
import common                                         # gemma_training/common.py: the training prompt
if not common.TAG:                                    # no data/config.json (always so in Docker): use HG_TAG_MODE
    common.TAG = "<hg>"
    common.TAG_MODE = "" if TAG_MODE == "none" else TAG_MODE
    if TAG_MODE in ("trigger", "before"):
        common.INSTRUCTION = ""                       # same as common.py does for these modes
from common import strip_tags
from homograph_lookup import load_words, fake_model, fake_chat
from prepare_data import nfc
from predict_batch import replace_tagged              # sahajBERT; does not need torch
from src.common import TAG, locate_tagged
from src import config as sahaj_config

GEMMA_HG = "Gemma 3 270M (<hg> = homograph, no tag = normal chat)"
SAHAJ = "sahajBERT homograph classifier"
CHOICES = [GEMMA_HG, SAHAJ]


def adapter_found():
    return (GEMMA_ADAPTER / "adapter_config.json").exists()


def sahaj_found():
    return (SAHAJ_MODEL / "config.json").exists()


MISSING_ADAPTER = (f"Gemma adapter not found in {GEMMA_ADAPTER}. Copy your trained gemma_cpu_lora folder "
                   "into the models folder, then restart (docker compose restart).")
MISSING_SAHAJ = (f"sahajBERT model not found in {SAHAJ_MODEL}. Copy the trained sahajbert/model folder "
                 "into the models folder as models/sahajbert, then restart (docker compose restart).")


class Gemma:
    """Loads Gemma once, on first use. With the adapter: rewrite() uses it, chat() switches it off.
    Without the adapter folder only chat() works (plain Gemma)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.has_adapter = False

    def load(self):
        with self.lock:
            if self.ready:
                return
            if FAKE:
                _, said = load_words()
                self._fake_rewrite, self.has_adapter = fake_model(said), True
                self.ready = True
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(os.cpu_count() or 4)
            self.torch = torch
            self.has_adapter = adapter_found()
            base = GEMMA_BASE
            if self.has_adapter and (GEMMA_ADAPTER / "base_model.txt").exists():
                base = (GEMMA_ADAPTER / "base_model.txt").read_text().strip()
            print(f"loading Gemma base model {base} (downloaded once, then cached)...", flush=True)
            self.tok = AutoTokenizer.from_pretrained(base)
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
            if self.has_adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(GEMMA_ADAPTER))
            self.model = model.eval()
            self.ready = True
            print("Gemma ready" + (" with the homograph adapter" if self.has_adapter else " (no adapter found)"), flush=True)

    def _generate(self, msgs, adapter_on, **kw):
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        off = self.model.disable_adapter() if (self.has_adapter and not adapter_on) else contextlib.nullcontext()
        with self.torch.no_grad(), off:
            out = self.model.generate(**enc, do_sample=False, **kw)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def rewrite(self, tagged):
        """Fine-tuned model, same prompt and settings as homograph_lookup.load_model."""
        self.load()
        if FAKE:
            return strip_tags(self._fake_rewrite(tagged))
        return strip_tags(self._generate(common.messages(tagged), True, max_new_tokens=128))

    def chat(self, text):
        """Plain Gemma (adapter off), same settings as homograph_lookup.load_model."""
        self.load()
        if FAKE:
            return fake_chat(text)
        return self._generate([{"role": "user", "content": text}], False, max_new_tokens=200, repetition_penalty=1.1)


class FakeSahaj:
    """STAND-IN for sahajBERT (HG_FAKE=1): always picks reading 1 from the lexicon."""

    def __init__(self, lexicon):
        self.lexicon = lexicon

    def predict(self, sentences, batch_size=None):
        out = []
        for s in sentences:
            m = self.lexicon[locate_tagged(s)]["1"]
            out.append({"pronunciation_variant": 1, "bangla_phoneme": m["bangla_phoneme"]})
        return out


class Sahaj:
    def __init__(self):
        self.lock = threading.Lock()
        self.predictor = None
        self.cfg = sahaj_config.load(verbose=False)
        import json
        self.lexicon = json.loads(Path(self.cfg["lexicon"]).read_text(encoding="utf-8"))

    def get(self):
        with self.lock:
            if self.predictor is None:
                if FAKE:
                    self.predictor = FakeSahaj(self.lexicon)
                else:
                    from predict_sentence import Predictor
                    self.predictor = Predictor(str(SAHAJ_MODEL), self.cfg["lexicon"])
            return self.predictor

    def check(self, sentence):
        """-> error text, or '' when the sentence is fine for sahajBERT."""
        try:
            word = locate_tagged(sentence)
        except ValueError as e:
            return str(e)
        if word not in self.lexicon:
            return f"'{word}' is not one of the sahajBERT words: {', '.join(self.lexicon)}"
        return ""


gemma = Gemma()
sahaj = Sahaj()


_TAGGED_WORD = re.compile(re.escape(TAG) + r"([\u0980-\u09ff]+)")


def gemma_prompt(sentence):
    """The model must see the tag the way it was trained. trigger mode (run.py default): the tag is one separate
    token at some gap of the sentence, so '<hg>কেন সে ...' becomes '<hg> কেন সে ...'. before mode: the tag
    stays glued to the word, as typed."""
    if common.TAG_MODE != "trigger":
        return sentence
    plain = _TAGGED_WORD.sub(lambda m: m.group(1), sentence).replace(TAG, " ")
    return TAG + " " + " ".join(plain.split())


def keep_homograph(source, answer):
    """The model sometimes drops the tagged word. If, for a tagged word of the 8, neither the word nor one of its
    respellings is in the answer, rebuild the sentence by hand with reading 1. -> (answer, note or '')"""
    lost = [w for w in _TAGGED_WORD.findall(source)
            if w in _LEX and not any(re.search("(?<![\u0980-\u09ff])" + re.escape(f) + "(?![\u0980-\u09ff])", answer)
                                     for f in [w] + [v["bangla_phoneme"] for v in _LEX[w].values()])]
    if not lost:
        return answer, ""
    fixed = source
    for w in _TAGGED_WORD.findall(source):
        fixed = fixed.replace(TAG + w, _LEX[w]["1"]["bangla_phoneme"] if w in _LEX else w, 1)
    return " ".join(fixed.replace(TAG, " ").split()), (
        "the model dropped the word (" + ", ".join(lost) + "), so reading 1 from the word list was used instead; "
        "reading 2 is not chosen by this fallback")


def run_gemma(lines):
    """One Gemma system, decided per line: <hg> in the line -> homograph adapter converts it;
    no <hg> -> the adapter is switched off and plain Gemma answers as usual."""
    rows = []
    for s in lines:
        if TAG in s:
            if not adapter_found() and not FAKE:
                rows.append([s, s, "", MISSING_ADAPTER])
                continue
            answer, warn = keep_homograph(s, gemma.rewrite(gemma_prompt(s)))
            rows.append([s, s, answer, "<hg> found: homograph adapter converted it" + (". Note: " + warn if warn else "")])
        else:
            rows.append([s, s, gemma.chat(s), "no <hg>: adapter off, plain Gemma answered normally"])
    return rows


def run_sahaj(lines):
    rows = [None] * len(lines)
    todo = []
    for i, s in enumerate(lines):
        if TAG not in s:
            rows[i] = [s, s, s, "no <hg>: returned unchanged (sahajBERT only resolves tagged homographs)"]
        elif not sahaj_found() and not FAKE:
            rows[i] = [s, s, "", MISSING_SAHAJ]
        elif err := sahaj.check(s):
            rows[i] = [s, s, "", "skipped: " + err]
        else:
            todo.append(i)
    if todo:
        batch = [lines[i] for i in todo]
        results = sahaj.get().predict(batch, sahaj.cfg["batch_size"])   # the whole list in one call
        for i, r in zip(todo, results):
            note = f"reading {r['pronunciation_variant']}"
            rows[i] = [lines[i], lines[i], replace_tagged(lines[i], r["bangla_phoneme"]), note]
    return rows


def run(text, model):
    lines = [nfc(l.strip()) for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return "", "", [], "Type a sentence first (or several, one per line)."
    try:
        rows = run_sahaj(lines) if model == SAHAJ else run_gemma(lines)
    except Exception as e:                          # show the problem on the page instead of a blank error
        return "", "", [], f"Error: {type(e).__name__}: {e}"
    return ("\n".join(r[2] for r in rows), pretty(rows), rows,
            f"{len(rows)} sentence(s) done with: {model}")


# ---- homograph words: the 8-word list, bolding, samples ----
from src.common import TAG as _T
_LEX = sahaj.lexicon                                 # the 8 words and their two respellings
HOMOGRAPHS = list(_LEX)
_BOLD = sorted({w for w in HOMOGRAPHS} | {v["bangla_phoneme"] for m in _LEX.values() for v in m.values()},
               key=len, reverse=True)
_BOLD_RE = re.compile("(?<![\u0980-\u09ff])(?:" + "|".join(map(re.escape, _BOLD)) + ")(?![\u0980-\u09ff])")


def bold(text):
    """HTML-safe text with every homograph word (or its respelling) in bold; the <hg> tag is shown in grey."""
    parts, last = [], 0
    for m in _BOLD_RE.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append("<b>" + html.escape(m.group(0)) + "</b>")
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts).replace(html.escape(_T), '<span style="opacity:.55">' + html.escape(_T) + "</span>")


def pretty(rows):
    out = ["<div style='line-height:1.9'>"]
    for i, r in enumerate(rows, 1):
        out.append(f"<p><span style='opacity:.6'>{i}. input:</span> {bold(r[1])}<br>"
                   f"<span style='opacity:.6'>&nbsp;&nbsp;&nbsp;output:</span> {bold(r[2]) if r[2] else '<i>(none)</i>'}<br>"
                   f"<span style='opacity:.6;font-size:.85em'>&nbsp;&nbsp;&nbsp;{html.escape(r[3])}</span></p>")
    out.append("</div>")
    return "".join(out)


SAMPLES = [
    "তুমি <hg>কেন এত দেরি করলে?",
    "দোকান থেকে নতুন কলম <hg>কেন।",
    "অনলাইনে কেনা ফোনটি <hg>আসল কিনা বুঝতে পারছি না।",
    "আজকের গানের <hg>আসর জমে উঠেছে।",
    "<hg>আসর নামাজের পরে আমি আসছি।",
    "এই পণ্যের দামে <hg>কর অন্তর্ভুক্ত আছে কি?",
    "কাঠমিস্ত্রি <hg>করাত দিয়ে কাঠ কাটছেন।",
    "গল্পের <hg>আসরে সবাই আজ আনন্দ করছে।",
    "<hg>আসরের আগে অর্ডারটি নিশ্চিত করে দিন।",
    "ছেলেটাকে সবাই <hg>কেবলা বলে ডাকে।",
]


def model_status():
    mark = lambda ok: "found" if ok else "**missing**"
    lines = [f"- Gemma adapter (`{GEMMA_ADAPTER}`): {mark(adapter_found())}",
             f"- sahajBERT model (`{SAHAJ_MODEL}`): {mark(sahaj_found())}",
             f"- Gemma base model: `{GEMMA_BASE}` (downloaded from Hugging Face the first time, then cached)"]
    if FAKE:
        lines.insert(0, "- **STAND-IN MODE (HG_FAKE=1): answers are not from the real models.**")
    return "\n".join(lines)


with gr.Blocks(title="Bangla homograph demo") as demo:
    gr.Markdown(
        "# Bangla homograph demo\n"
        f"Put `{TAG}` right before the homograph (no space), for example `তুমি {TAG}কেন এত দেরি করলে?`. "
        f"A sentence with `{TAG}` is converted by the homograph model (the tag is removed and the word is "
        "replaced by its pronunciation spelling). A sentence without the tag gets the model's normal behaviour. "
        "One sentence per line for many at once.\n\n" + model_status())
    with gr.Row():
        model = gr.Dropdown(CHOICES, value=GEMMA_HG, label="Model")
    gr.Markdown("**The 8 homograph words:** " + " &nbsp;·&nbsp; ".join(f"**{w}** ({_LEX[w]['1']['bangla_phoneme']} / {_LEX[w]['2']['bangla_phoneme']})" for w in HOMOGRAPHS))
    text = gr.Textbox(label="Sentence(s), one per line", lines=5)
    btn = gr.Button("Run", variant="primary")
    out = gr.Textbox(label="Output, one per line", lines=5, interactive=False)
    shown = gr.HTML(label="Result with the homograph words in bold")
    status = gr.Markdown()
    table = gr.Dataframe(headers=["input", "sent to the model", "output", "what happened"],
                         label="Details", interactive=False, wrap=True)
    gr.Examples([[x] for x in SAMPLES], inputs=[text], label="10 sample inputs (click one to fill the box)")
    btn.click(run, [text, model], [out, shown, table, status])

if __name__ == "__main__":
    print("\nOpen http://localhost:7860 in your browser.\n", flush=True)
    demo.queue().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                        server_port=int(os.environ.get("PORT", "7860")), share=SHARE)
