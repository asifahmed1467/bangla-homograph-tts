# Train Gemma 4 on the Bangla homograph data (beginner guide)

New to training? Read [HOW_TRAINING_WORKS.md](HOW_TRAINING_WORKS.md) first (Bangla version: [HOW_TRAINING_WORKS_bn.md](HOW_TRAINING_WORKS_bn.md)). It explains every step below in plain language.

**What the model learns:** you give it a normal sentence, and it gives the same sentence back with the homograph written in its pronunciation spelling.

```
input : ভিডিও কলে আজ গল্পের আসর বেশ জমে উঠেছে।
output: ভিডিও কলে আজ গল্পের আশর বেশ জমে উঠেছে।

input : আমি আজ সকালে বাজারে গিয়ে কিছু সবজি কিনেছি।    (no homograph)
output: আমি আজ সকালে বাজারে গিয়ে কিছু সবজি কিনেছি।    (unchanged)
```

The model only knows what the examples show it. That is why the training data mixes homograph sentences with plain sentences that come back unchanged.

## Code check on your own PC (no GPU, no Colab): start here

This checks that the fine-tuning **code** works from start to finish on a normal Windows PC (CPU only, 16 GB RAM). It uses tiny data and 10 training steps, so it finishes in a few minutes. It does **not** make a good model; that needs the full run.

**One-time setup (Windows 11, PowerShell)**
1. Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/). On the first installer screen, tick **"Add python.exe to PATH"**.
2. Unzip `gemma_training.zip`, for example to `F:\gemma`. You get `F:\gemma\gemma_training`, `F:\gemma\tts_readiness` and `F:\gemma\homograph_call_context`. Keep them side by side.
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
2. scores the original model on 8 test sentences;
3. fine-tunes it with LoRA for 10 steps;
4. reloads the saved adapter and scores it again;
5. writes a before/after table (`smoke_comparison.md`).

**What "working" looks like.** The end of the output shows:
```
================ SMOKE TEST RESULT ================
[PASS] 1. Prepare tiny dataset
[PASS] 2. Score the ORIGINAL model
[PASS] 3. Fine-tune (LoRA) for a few steps
[PASS] 4. Reload the saved adapter and score it
[PASS] 5. Before/after table
[PASS] data files written
[PASS] adapter files saved
[PASS] loss went down (x.xx -> y.yy)
[PASS] predictions file written (8 sentences)
====================================================
ALL CHECKS PASSED: the fine-tuning code works on this PC.
```
In plain words, the code works when:
- every step finishes without a red error;
- the training **loss goes down** (the model is learning from your data);
- the adapter files are saved in `smoke_lora/` (`adapter_config.json` and `adapter_model.safetensors`);
- the saved adapter loads again and writes predictions to `smoke_after.csv`.

The scores in the table after 10 steps are **not** meaningful yet. If a line says FAIL, copy the error text just above it and send it.

**After the check: a real (small) training on the CPU**
```
python prepare_data.py --words 12
python evaluate_cpu.py --out predictions_before.csv
python train_cpu.py
python evaluate_cpu.py --adapter gemma_cpu_lora --out predictions.csv
python compare.py
```
This takes roughly 15 to 40 minutes on a CPU and shows whether the model really learns. For a stronger model, add `--model google/gemma-3-1b-it` to the `evaluate_cpu.py` (first one) and `train_cpu.py` lines. That one needs a free Hugging Face account: accept the licence on its page, then run `hf auth login`.

**What fits on a CPU with 16 GB RAM**

| Model | Fits? | Note |
|---|---|---|
| Gemma 4 E2B (main pipeline) | No, needs about 20 GB to train on CPU | needs a GPU |
| Gemma 3 270M (default here) | Yes, about 4 GB | fast enough on a CPU, weaker Bangla |
| Gemma 3 1B | Yes, but tight | about 4 times slower |

## Quick test on Google Colab (needs a Google account)

This trains on a small part of the data, so the whole check takes well under an hour.

**A. On Google Colab (training, because a 4 GB GPU is too small)**
1. Go to [colab.research.google.com](https://colab.research.google.com), choose **File → Upload notebook**, and pick `colab_train.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order. The notebook is already in QUICK TEST mode:
   - step 4 picks 12 homograph words (about 240 sentences) plus 200 plain sentences, and keeps 15% aside as unseen test sentences;
   - step 5 scores the **original** Gemma on those test sentences;
   - step 6 fine-tunes for 2 epochs (a few minutes);
   - step 7 scores the **fine-tuned** model on the same sentences;
   - step 8 (`compare.py`) prints the before/after table and saves it as `comparison.md`.
4. Steps 10 and 11 make the GGUF file for your PC and download it with `result.zip`.

**How to read the table.** Fine-tuning worked if the fine-tuned column is clearly higher than both *Original Gemma* and *Change nothing* on homograph sentences, and plain sentences stay near 100%. *Change nothing* is a model that just copies the input. About half the readings are spelled like the input, so copying already scores some points.

**B. On your own PC (running the model)**
1. Unzip `result.zip` into this `gemma_training` folder. It holds `data/` with the test sentences and your settings.
2. Install llama.cpp: `winget install llama.cpp`
3. Start your fine-tuned model: `llama-server -m path\to\your-file.gguf -ngl 99 -c 1024` (out of memory on 4 GB? use `-ngl 20` or `-ngl 0`).
4. In a second terminal, score it locally: `python run_local.py --test data/test.jsonl --out local_after.csv`
5. Optional, local "before": stop the server (Ctrl+C), start the original model with `llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M -ngl 99 -c 1024`, and run `python run_local.py --test data/test.jsonl --out local_before.csv`.
6. `python compare.py --before local_before.csv --after local_after.csv` shows the same before/after table, now from your own PC.

When the quick test looks good, run the FULL lines in notebook steps 4 and 6.

## Try the trained model: web page and example tests

Both use the adapter folder `gemma_cpu_lora` made by `train_cpu.py`. It is not in this repo (it only exists on the PC that trained it), so copy that folder into `gemma_training/` next to `app.py`, or pass its path with `--adapter`.

They follow the plan: look for a word from `my_words_8.csv` in the sentence, put `<hg>` directly in front of it, and call the model **only** if one was found. A sentence with no listed word comes back unchanged and the model is not called. The lookup is in `homograph_lookup.py`, shared by both.

**Windows, step by step** (open Command Prompt):
```bat
cd gemma_training
py -m venv venv
venv\Scripts\activate
pip install -r requirements_cpu.txt

python app.py
```
Then open http://127.0.0.1:7860 in your browser, type a sentence and press Run. The page shows the input sent to the model, the output, and whether the model was called. Stop it with Ctrl+C.

```bat
python test_examples.py
```
This runs 22 example sentences and prints PASS or FAIL for each, then a summary. Twelve come from the training CSV; ten are new sentences whose expected answers are a best guess, so a FAIL there can be a wrong expectation (this is written at the top of the file).

Other computers on the same Wi-Fi: `python app.py --lan` prints an address like http://192.168.x.x:7860 for them to open (allow Python in the Windows firewall prompt).

If no listed word is found, the page switches the adapter off and plain Gemma 3 270M answers like a normal chat (the tiny model can give weak answers). The page and `python test_examples.py --sentence "..."` both do this; the 22-example table checks only the rewriting path.

Options: `--adapter D:\path\to\gemma_cpu_lora` (adapter in another place), `python test_examples.py --lookup-only` (no model, checks only which sentences reach the model), `--fake` on either script (a stand-in that is **not** Gemma, only to check the page and script without an adapter). The first run downloads the base model `unsloth/gemma-3-270m-it` (needs internet), and if `data/config.json` is missing it assumes the `<hg>` tag was used with `--tag_mode before`.

The trained words are only: আসর, আসরে, আসরের, আসল, কর, করাত, কেন, কেবলা. Other homographs are not looked up.

## What you need

| Thing | Needed |
|---|---|
| GPU | NVIDIA, 8 GB VRAM or more (RTX 3060/4060 and up). 16 GB+ lets you use the bigger E4B model. **4 GB cards such as the GTX 1050 Ti are too small**, so use Colab. |
| OS | Linux, or Windows with **WSL2** (Ubuntu). WSL2 is the easiest route on Windows. |
| Python | 3.10 to 3.12 |
| Disk | about 20 GB free (model download + checkpoints) |
| Hugging Face account | free; accept the Gemma licence on the model page once, then run `hf auth login` |

Less than 8 GB of VRAM, or no NVIDIA GPU? Open `colab_train.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook). It runs every step below on a free T4 GPU (15 GB), and you only need a browser.

## The 4 steps

### Step 1. Set up Python (once)

```bash
python -m venv gemma-env
source gemma-env/bin/activate            # on Windows PowerShell: gemma-env\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"    # must print True
```

### Step 2. Build the training files

Unzip `gemma_training.zip`. It already holds the scripts and the data folders in the right layout. Go into the `gemma_training` folder, then:

```bash
python prepare_data.py --wiki 1500
```

This reads `tts_readiness/train_ready_rows.csv` (1,880 checked sentences, 95 words), keeps 15% of every word's sentences aside for testing, and adds 1,500 plain sentences from Bangla Wikipedia that contain no homograph. You get `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl`.
Want a tag in the sentence (for example `<hg>`)? Add `--tag "<hg>" --tag_mode trigger` (tag anywhere), `--tag_mode before` (tag right before the homograph, e.g. `যাচ্ছ <hg>কেন?`) or `--tag_mode mark` (tag around the homograph word). Avoid `. , ! / @` as tags. Full explanation in Bangla: [TAG_GUIDE_bn.md](TAG_GUIDE_bn.md).
Have your own plain sentences? Use `--plain_file my_sentences.txt` (one per line) instead of, or as well as, `--wiki`.

### Step 3. Train

```bash
python train.py
```

It downloads Gemma 4 E2B, loads it in 4-bit, and trains small LoRA layers on top. The original model is not changed, so it keeps its general Bangla knowledge. Expect roughly 15 to 40 minutes on a mid-range GPU. The result is a small folder `gemma_homograph_lora/`.

Watch the printed `eval_loss`. It should go down. If it goes up again in the last epoch, set `num_train_epochs=2`.

### Step 4. Test

```bash
python evaluate.py
python evaluate.py --sentence "আসর নামাজের পর আমরা বাজারে যাব।"
```

It prints two numbers. Both matter:
- **homograph sentences fixed correctly**: the model chose the right reading.
- **plain sentences left unchanged**: the model did not damage normal text.

Every prediction is saved in `predictions.csv`, so you can see which words it gets wrong.
Tip: run `python evaluate.py --adapter unsloth/gemma-4-E2B-it` once before training. That scores the untrained model, so you can see how much training helped.

## If something goes wrong

| Problem | Fix |
|---|---|
| `CUDA out of memory` | In `train.py` set `per_device_train_batch_size=4` (or 2) and `gradient_accumulation_steps=4` (or 8). |
| `torch.cuda.is_available()` prints False | The NVIDIA driver or the CUDA build of torch is missing. Reinstall torch with the `--index-url` line above. |
| 401 / gated repo error | Accept the Gemma licence on Hugging Face and run `hf auth login`. |
| Model changes words it should not | Add more plain sentences (`--wiki 3000`). |
| Model is wrong on one word | That word needs more and more varied sentences (see `tts_readiness/word_status.csv`, column `contexts_to_add`). |

## Good to know

- **The model copies your labels.** If a sentence in the CSV has the wrong reading, the model learns the mistake. 40 words are still marked for a native-speaker check in `word_status.csv`.
- **It will not learn words that are not in the data.** A homograph it never saw will most likely be left unchanged.
- **Only the 95 TRAIN_READY words are used.** When you fix more words, regenerate `train_ready_rows.csv` or point `--csv` at your own file with the columns `homograph, pronunciation_variant, sentence_homograph_form, sentence_tts`.
- The settings for model name and sequence length are in `common.py`.

## Run the trained model on your own PC (works on a GTX 1050 Ti)

Running needs much less memory than training. The Colab notebook's step 9 runs `export_gguf.py`, which makes one file of about 3 GB in the GGUF format, and step 10 downloads it. Then, on your PC:

1. Install llama.cpp: `winget install llama.cpp` (Windows) or `brew install llama.cpp` (Mac).
2. Start the model: `llama-server -m path\to\your-file.gguf -ngl 99 -c 1024`
   - `-ngl 99` puts the whole model on the GPU. If you get *out of memory* on a 4 GB card, use `-ngl 20` (part GPU, part CPU) or `-ngl 0` (CPU only, slower but works).
3. In a second terminal, from this folder: `python run_local.py "আসর নামাজের পর আমরা বাজারে যাব।"`, or `python run_local.py --file sentences.txt` for a whole file.

You can also open http://127.0.0.1:8080 in a browser and chat with it there. Paste the instruction from `common.py` before the sentence, because the model was trained with it.
