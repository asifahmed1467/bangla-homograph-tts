# Train Gemma 3 270M on the Bangla homograph data (CPU, beginner guide)

New to training? Read [HOW_TRAINING_WORKS.md](HOW_TRAINING_WORKS.md) first (Bangla version: [HOW_TRAINING_WORKS_bn.md](HOW_TRAINING_WORKS_bn.md)). It explains every step below in plain language.

**What the model learns:** you give it a normal sentence, and it gives the same sentence back with the homograph written in its pronunciation spelling.

```
input : ভিডিও কলে আজ গল্পের আসর বেশ জমে উঠেছে।
output: ভিডিও কলে আজ গল্পের আশর বেশ জমে উঠেছে।

input : আমি আজ সকালে বাজারে গিয়ে কিছু সবজি কিনেছি।    (no homograph)
output: আমি আজ সকালে বাজারে গিয়ে কিছু সবজি কিনেছি।    (unchanged)
```

The model only knows what the examples show it. That is why the training data mixes homograph sentences with plain sentences that come back unchanged.

## Code check on your own PC: start here

This checks that the fine-tuning **code** works from start to finish on a normal Windows PC (CPU only, 16 GB RAM). It uses tiny data and 10 training steps, so it finishes in a few minutes. It does **not** make a good model; that needs the full run.

**One-time setup (Windows 11, PowerShell)**
1. Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/). On the first installer screen, tick **"Add python.exe to PATH"**.
2. Get the repo (for example `C:\Users\Reve\merged_repo`). `gemma_training`, `tts_readiness` and `homograph_call_context` must stay side by side.
3. Open the `gemma_training` folder in File Explorer, click the address bar, type `powershell` and press Enter.
4. Create an environment and install the libraries (about 5 to 10 minutes, about 1 GB download):
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements_cpu.txt
   ```
   You should now see `(venv)` at the start of the line. If `activate` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and run it again.

**Run the check**
```
python smoke_test.py
```
The first run downloads Gemma 3 270M (about 0.6 GB, no Hugging Face account needed). Then it runs these steps by itself:
1. builds a tiny dataset (3 homograph words + 24 plain sentences, no internet needed);
2. fine-tunes it with LoRA for 10 steps.

**What "working" looks like.** The end of the output shows:
```
================ SMOKE TEST RESULT ================
[PASS] 1. Prepare tiny dataset
[PASS] 2. Fine-tune (LoRA) for a few steps
[PASS] data files written
[PASS] adapter files saved
[PASS] loss went down (x.xx -> y.yy)
====================================================
ALL CHECKS PASSED: the fine-tuning code works on this PC.
```
In plain words, the code works when:
- every step finishes without a red error;
- the training **loss goes down** (the model is learning from your data);
- the adapter files are saved in `smoke_lora/` (`adapter_config.json` and `adapter_model.safetensors`).

10 steps are far too few for a good model; this only shows the code runs. If a line says FAIL, copy the error text just above it and send it.

**After the check: a real (small) training on the CPU**
```
python prepare_data.py --words 12
python train_cpu.py
python test_examples.py --adapter gemma_cpu_lora
```
This takes roughly 15 to 40 minutes on a CPU. For a stronger model, add `--model google/gemma-3-1b-it` to the `train_cpu.py` line. That one needs a free Hugging Face account: accept the licence on its page, then run `hf auth login`.

**What fits on a CPU with 16 GB RAM**

| Model | Fits? | Note |
|---|---|---|
| Gemma 3 270M (default here) | Yes, about 4 GB | fast enough on a CPU, weaker Bangla |
| Gemma 3 1B | Yes, but tight | about 4 times slower |

## Try the trained model

The web page for both models is `web_app.py` in the top folder (with or without Docker, see `README_DOCKER.md`). From the command line:

```
python test_examples.py --adapter gemma_cpu_lora
python test_examples.py --adapter gemma_cpu_lora --sentence "আসর নামাজের পর আমরা বাজারে যাব।"
```
The first line runs 22 example sentences and prints PASS or FAIL for each. Twelve come from the training CSV; ten are new sentences whose expected answers are a best guess, so a FAIL there can be a wrong expectation. The second line tries one sentence of your own.

Both look for a word from `my_words_8.csv`, put `<hg>` directly in front of it, and call the model **only** if one was found (`homograph_lookup.py`). With no listed word, plain Gemma 3 270M answers like a normal chat. `--lookup-only` checks only the word lookup, without a model.

The trained words are only: আসর, আসরে, আসরের, আসল, কর, করাত, কেন, কেবলা. Other homographs are not looked up.

## Build the training files yourself

Go into the `gemma_training` folder, then:

```
python prepare_data.py --wiki 1500
```

This reads `tts_readiness/train_ready_rows.csv` (1,880 checked sentences, 95 words), keeps 15% of every word's sentences aside for testing, and adds 1,500 plain sentences from Bangla Wikipedia that contain no homograph. You get `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl`.
Want a tag in the sentence (for example `<hg>`)? Add `--tag "<hg>" --tag_mode trigger` (tag anywhere), `--tag_mode before` (tag right before the homograph, e.g. `যাচ্ছ <hg>কেন?`) or `--tag_mode mark` (tag around the homograph word). Avoid `. , ! / @` as tags. Full explanation in Bangla: [TAG_GUIDE_bn.md](TAG_GUIDE_bn.md).
Have your own plain sentences? Use `--plain_file my_sentences.txt` (one per line) instead of, or as well as, `--wiki`.

For the two-tag data, use `python prepare_two_tag.py` instead (see `TWO_TAG_UPDATE.md` in the top folder).

## If something goes wrong

| Problem | Fix |
|---|---|
| Out of memory while training | Use a smaller batch: `python train_cpu.py --batch 2` (or set `HG_GEMMA_BATCH=2`). |
| 401 / gated repo error | Only for `google/...` models: accept the Gemma licence on Hugging Face and run `hf auth login`. The default `unsloth/gemma-3-270m-it` needs no login. |
| Model changes words it should not | Add more plain sentences (`--wiki 3000`). |
| Model is wrong on one word | That word needs more and more varied sentences (see `tts_readiness/word_status.csv`, column `contexts_to_add`). |

## Good to know

- **The model copies your labels.** If a sentence in the CSV has the wrong reading, the model learns the mistake. 40 words are still marked for a native-speaker check in `word_status.csv`.
- **It will not learn words that are not in the data.** A homograph it never saw will most likely be left unchanged.
- **Only the 95 TRAIN_READY words are used.** When you fix more words, regenerate `train_ready_rows.csv` or point `--csv` at your own file with the columns `homograph, pronunciation_variant, sentence_homograph_form, sentence_tts`.
- The prompt and tag settings are in `common.py`. The web app reads the same file, so keep them in step with the adapter you trained.
