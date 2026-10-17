# Bangla Homograph Pronunciation — sahajBERT CPU

Fine-tune the **Apache-2.0** [sahajBERT](https://huggingface.co/neuropark/sahajBERT) ALBERT encoder for **binary pronunciation-variant selection**, then map the predicted variant to Latin and Bengali phoneme fields. Runs with CPU only; weights download on first training run.

## Windows PowerShell — from project folder

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python prepare.py
python train.py --epochs 5 --batch-size 8
python predict.py --sentence "তুমি <hg>কেন এত দেরি করলে?" --output result.json
python predict.py --input data/sample_input.json --output results.csv
```

If your venv is called `venv`, use `venv\Scripts\Activate.ps1` instead. If PowerShell script execution is blocked, use `cmd.exe` with `.venv\Scripts\activate.bat`.

## Tag contract

- At inference, supply **exactly one** `<hg>` token immediately before the target homograph, e.g. `তুমি <hg>কেন এত দেরি করলে?`.
- Do **not** write `<hg> কেন` (no space after tag).
- The input word MUST be the **canonical spelling** from dataset `homograph` column.
- For multiple tagged homographs in one sentence, submit one copy per target, each with only one tag.
- The supplied CSV does **not** contain `বস`. To add that word, extend the dataset with correctly labeled `বস` training rows and retrain.
- The training loader automatically replaces the training dataset's pronunciation-specific surface forms with canonical spelling and inserts `<hg>`.
- The tag is added to the tokenizer vocabulary and the encoder embeddings are resized. The newly added embedding is initialized during fine-tuning.

## Output contract

A JSON array / CSV table with fields: `sentence`, `homograph`, `pronunciation_variant` (1/2), `phoneme`, `bangla_phoneme`, `confidence`.

## Dataset audit — important

The supplied attachment contains **320 rows, 8 homographs, 20 rows per variant**. It includes a duplicate sentence and several near-duplicates/paraphrases. Many sentences use the *label spelling*, rather than the homograph (`ক্যানো` instead of `কেন`; `আশর` instead of `আসর`; `করো` instead of `কর`). `prepare.py` substitutes the matched target with the canonical word before training so it cannot directly cheat by inspecting spelling.

**Manual linguistic review is essential:** some `homograph`/`bangla_phoneme` pairs are different written forms rather than established same-spelling pronunciation alternatives (e.g. `আসল` and `আশলো`). The source also includes examples where pronunciation variants should be independently verified. Labels and transliterations are reproduced as given, not certified as phonetically correct. Before real-world deployment, validate your lexicon with Bengali linguists/TTS specialists.

The validation split groups exact canonicalized sentences but **does not separate near-paraphrases**. Scores from this dataset can be misleading. For trustworthy results, build a separate clean test set with genuinely new contexts for each word/pronunciation. Also test out-of-domain sentences and consider abstaining for low-confidence predictions.

This is a **two-class model** (output classes map to labels 1/2) trained across words. All training examples require exactly two variants and a consistent variant-labeling scheme. Some words not included here may require other handling. It does not generate general-purpose IPA for arbitrary Bangla words.

## Commercial use

Base model is Apache-2.0 (check its model card); comply with NOTICE/license requirements as applicable. Independently review your dataset, packages and downstream usage rights. This project source is MIT-licensed, except supplied data remains under its original owner/licensing terms.

## Important

Training has **not** been executed here because the model must be downloaded and CPU training is environment-dependent. `prepare.py` can be run offline. If Bengali rendering is broken in the terminal, pass `--output results.csv` and open it in VS Code.

## Settings and menu (config.json, HG_* env vars, run.py)

Defaults live in `config.json` (model_name, raw_data, processed_data, lexicon, model_dir, epochs, batch_size, lr, max_length, seed, val_ratio, threads). Priority: CLI flag > environment variable `HG_<KEY>` > `config.json`. Values taken from the environment are printed to stderr as `[config] ...`.

Easiest way: `python run.py` and pick 1 (prepare), 2 (train), 3 (type a tagged sentence), 4 (show config), 0 (quit).

Windows cmd:

```bat
set HG_EPOCHS=2
python run.py
python train.py --batch-size 4
set HG_EPOCHS=
```

PowerShell:

```powershell
$env:HG_EPOCHS=2
python run.py
Remove-Item Env:HG_EPOCHS
```
