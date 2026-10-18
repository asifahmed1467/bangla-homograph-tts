# Bangla Homograph Pronunciation (Gemma and sahajBERT)

This repo holds two models. You pick one by configuration, not by editing code.

## Pick a model

- `gemma_training/` is the Gemma LoRA model (rewrites the sentence).
- `sahajbert/` is the sahajBERT classifier (picks pronunciation 1 or 2 for 8 words). It has its own `config.json` (epochs, batch size and so on) and its own data.
- Top-level `config.json` chooses the model:

```json
{ "model": "gemma" }
```

Change it to `"sahajbert"`, or override it with the environment variable `HG_MODEL` (it wins over the file). The sahajBERT settings can be overridden the same way with `HG_<KEY>` (for example `HG_EPOCHS`).

Easiest: run the menu from the repo folder and type 1 to choose the model.

```bat
py -m venv venv
venv\Scripts\activate
pip install -r gemma_training\requirements_cpu.txt
pip install -r sahajbert\requirements.txt
python run.py
```

One-time override in Windows cmd:

```bat
set HG_MODEL=sahajbert
python run.py
set HG_MODEL=
```

PowerShell: `$env:HG_MODEL="sahajbert"` and `Remove-Item Env:HG_MODEL`.

The two models use different data formats on purpose; each reads its own files. Each model folder still works on its own, as described below and in its own README.

---

## Gemma part

LoRA fine-tuning of Gemma 3 270M to rewrite Bangla sentences so homographs get the right pronunciation for TTS.

- `gemma_training/` holds the scripts and the full guides (`README.md`, `HOW_TRAINING_WORKS.md`, `TAG_GUIDE_bn.md`).
- `homograph_call_context/` and `tts_readiness/` hold the source CSVs that `prepare_data.py` reads, so keep them next to `gemma_training/`.

Generated data (`data/`), virtual environments and trained adapters (`gemma_cpu_lora/`, `smoke_lora/`) are not in the repo.

## Quick start (Windows, CPU)

```bat
cd gemma_training
py -m venv venv
venv\Scripts\activate
pip install -r requirements_cpu.txt

python smoke_test.py

python prepare_data.py --smoke --csv my_words_8.csv --test_frac 0.2 --tag "<hg>" --tag_mode trigger
python train_cpu.py --epochs 3
python evaluate_cpu.py --adapter gemma_cpu_lora --sentence "..."
```

Try the trained model in a browser with `python app.py`, or check example sentences with `python test_examples.py` (see `gemma_training/README.md`, section "Try the trained model"). Put your `gemma_cpu_lora` folder next to the scripts first.

For GPU training on Colab, see `gemma_training/colab_train.ipynb` and `gemma_training/README.md`.

## Two-tag update

Both models accept up to 2 `<hg>` tags per sentence, and the web app shows timing and supports batch input. See `TWO_TAG_UPDATE.md` for the data, the training commands and the evaluation steps.

## Settings (environment variables)

Every setting can be changed with an environment variable, so nobody has to edit code. Leave a variable out and its default is used, which is the behaviour described above.

**With Docker:** copy `.env.example` to `.env` next to `docker-compose.yml`, change the values you need, then `docker compose up`. Docker Compose reads `.env` by itself. With a saved image: `docker run --env-file .env -p 8080:8080 bangla-homograph:latest` (the `-p` port must match `HG_PORT`).

**Without Docker:** set the variable before running Python. cmd: `set HG_PORT=8080`, PowerShell: `$env:HG_PORT="8080"`.

| Variable | Default | What it does |
|---|---|---|
| `HG_PORT` | `7860` | Port of the web page |
| `HG_HOST` | `127.0.0.1` (Docker: `0.0.0.0`) | Address the web page listens on |
| `HG_DEFAULT_MODEL` | `gemma` | Model selected when the page opens: `gemma` or `sahajbert` |
| `HG_TAG_MODE` | `before` | How the Gemma adapter was trained: `before`, `trigger` or `none` |
| `HG_MAX_TAGS` | `2` | `<hg>` tags corrected per sentence (the models were trained with at most 2) |
| `HG_ASR_RULE` | `1` | `0` switches off the keyword safety net for আসর/আসরে/আসরের |
| `HG_ASR_WINDOW` | `4` | Words on each side of the tagged word the safety net looks at |
| `HG_SHARE` | `0` | `1` also prints a temporary public gradio.live link |
| `HG_FAKE` | `0` | `1` gives stand-in answers, only to check the page |
| `HG_MODELS_DIR` | `models/` (Docker: `/models`) | Folder holding `gemma_cpu_lora/` and `sahajbert/` |
| `HG_GEMMA_ADAPTER` | `<models>/gemma_cpu_lora` | Gemma LoRA adapter folder (also the default `--adapter` of `evaluate_cpu.py`) |
| `HG_SAHAJBERT_MODEL` | `<models>/sahajbert` | Trained sahajBERT folder (also the default `--model` of `sahajbert/evaluate_2tag.py`) |
| `HG_GEMMA_BASE` | `unsloth/gemma-3-270m-it` | Gemma base model when the adapter has no `base_model.txt` (not used in Docker, where the base is baked in) |
| `HG_GEMMA_EPOCHS` | `2` | Default `--epochs` of `gemma_training/train_cpu.py` |
| `HG_GEMMA_BATCH` | `4` | Default `--batch` of `train_cpu.py` |
| `HG_GEMMA_OUT` | `gemma_cpu_lora` | Default `--out` folder of `train_cpu.py` |
| `HG_MODEL` | from `config.json` | `run.py` model choice: `gemma` or `sahajbert` |
| `HG_<KEY>` | `sahajbert/config.json` | Overrides any sahajBERT setting, e.g. `HG_EPOCHS`, `HG_BATCH_SIZE`, `HG_LR`, `HG_MAX_LENGTH`, `HG_THREADS` |

Command-line flags still win over the environment variables in the training and evaluation scripts. In Docker the models are baked into the image, so changing model folders needs a rebuild (or a mounted folder).
