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
