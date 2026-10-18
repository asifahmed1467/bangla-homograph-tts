# UPDATE v3 (আসর fix: keyword safety net + 106 new training rows)
- Problem seen: the আসর family (আসর, আসরে, আসরের) was mostly wrong at the start of a sentence; the models lean on the word just before the tag.
- `gemma_training/two_tags.py` has `asr_rule`: cue words within 4 words of a tagged আসর/আসরে/আসরের (same sentence) decide the reading when only one kind is present. Prayer cues: নামাজ, ওয়াক্ত, আজান, জামাত, মসজিদ, অজু, মুসল্লি, ইমাম, মুয়াজ্জিন, রাকাত, কাজা, or পর/পরে/আগে/সময় right after the word. Gathering cues: গান, গজল, কবিতা, আড্ডা, জমে/জমাতে, মঞ্চ, শিল্পী, বসেছিল/বসল/বসবে, ভাঙল, জলসা, মেলার, গল্প, বাউল, শ্রোতা, পুঁথি, তবলা and similar. Both kinds or none: the model decides.
- `web_app.py` applies it to sahajBERT and Gemma answers; the "what happened" column says "keyword rule: ... (because of '...')" when it changed a word.
- Checked: all 8 আসর words in the user's last results CSV come out right; on the existing 2-tag train+test rows the rule fired 76 times and agreed with the label every time.
- `two_tag_train.csv` now has 311 rows (205 v2 + 106 new: 90 আসর-family rows, 15 per form and reading, half sentence-initial, prayer mostly without নামাজ right next to it; plus 16 extra কেবলা/কেন pairs). `two_tag_test.csv` is unchanged.
- Retrain: Gemma `python prepare_two_tag.py` then `python train_cpu.py --epochs 7 --out gemma_cpu_lora_2tag_v3`; sahajBERT `python prepare_2tag.py` then `python train.py --data data/processed_2tag.csv --epochs 7 --batch-size 8 --out model_2tag_v3`.

# UPDATE v2 (more varied data, 7 epochs)
- `two_tag_train.csv` now has **205** two-tag sentences (103 from v1 + 102 new, in more varied shapes, 95 call-center / e-commerce style).
- `two_tag_test.csv` now has **30** held-out sentences: the 20 from v1 + 10 hand-made ones (ids `th001` to `th010`) that were checked against all project files. Never trained on. Old scores (20 sentences) and new scores (30 sentences) are not directly comparable.
- Defaults changed: 16 two-tag sentences go to validation.
- Epochs: the user asked for 7 (from 5). Commands: Gemma `python train_cpu.py --epochs 7 --out gemma_cpu_lora_2tag_v2`; sahajBERT `python train.py --data data/processed_2tag.csv --epochs 7 --batch-size 8 --out model_2tag_v2`.
- Expect: more data and varied shapes should help most with the "why"/"buy" (কেন) mistakes. More epochs on a small set can also overfit, so compare the held-out scores of v1 and v2 before replacing the Docker models.
- Please have a native speaker check the readings in both CSVs.

---

# Two-tag homograph update (Gemma + sahajBERT, same Docker app)

## What changes
- Both models: a sentence can carry **up to 2 `<hg>` tags**, both words are respelled. From a **3rd `<hg>`** on, the rest of the sentence comes back as typed, but without the `<hg>` tags (those words are not respelled). Enforced in code (`gemma_training/two_tags.py`), so it holds whatever the models do. One-tag and no-tag sentences work as before.
- sahajBERT still classifies one word at a time: a 2-tag sentence is sent as 2 one-tag copies in the same batch call.
- **Time:** every sentence on the page shows how long it took (ms), plus total and average. Model loading on the first run is shown separately. The test scripts print the time too.
- **Batch:** paste many lines (one sentence per line) and download all results as a CSV. There is no file-upload box.
- Same Docker app, same Dockerfile and docker-compose.yml, same model folders (`models\gemma_cpu_lora`, `models\sahajbert`).

## Files and where they go in `C:\Users\Reve\merged_repo`
| From the zip | To | New or changed |
|---|---|---|
| `web_app.py` | `merged_repo\` | changed (back up first) |
| `gemma_training\common.py` | `merged_repo\gemma_training\` | changed (back up first) |
| `gemma_training\two_tags.py`, `prepare_two_tag.py`, `two_tag_train.csv`, `two_tag_test.csv` | `merged_repo\gemma_training\` | new |
| `sahajbert\prepare_2tag.py` | `merged_repo\sahajbert\` | new |

## Data
- `two_tag_train.csv`: 103 sentences, each with 2 tagged words from the 8, both readings mixed; 49 call-center / e-commerce style (GP, Airtel, Amber IT, online orders), 54 general.
- `two_tag_test.csv`: 20 more two-tag sentences, only for testing, never trained on.
- Old CSVs are not changed; old one-tag and no-tag rows stay in the training mix. No sentence repeats one already in the project.
- Gemma mix: train 292 (95 two-tag, 179 one-tag, 18 no-tag), val 25, test 74 (20 two-tag).
- sahajBERT mix: old 255 train / 64 val rows, plus 190 train / 16 val rows from the two-tag sentences (2 rows per sentence).

## Honest expectation
About 100 two-tag sentences is small. Expect roughly a third to two thirds of the 20 test sentences fully right for Gemma; per-word accuracy higher. sahajBERT is usually better at picking the reading, since it only chooses 1 or 2.

## Please check
The sentences were written by Claude. A native speaker should check each sentence and its readings (`reading_1`, `reading_2`).
Readings: আসর/আসরে/আসরের 1 = gathering (আশর…), 2 = Asr prayer · আসল 1 = genuine (আশল), 2 = came (আশলো) · কর 1 = tax, 2 = do! (করো) · করাত 1 = saw, 2 = made someone do (করাতো) · কেন 1 = why (ক্যানো), 2 = buy! (কেনো) · কেবলা 1 = qibla, 2 = silly (ক্যাবলা)

## Steps (Windows cmd, venv on)
1. Back up, copy files in (see table).
2. `cd gemma_training` → `python prepare_two_tag.py`
3. `python train_cpu.py --epochs 7 --out gemma_cpu_lora_2tag_v3`
4. `cd ..\sahajbert` → `python prepare.py` → `python prepare_2tag.py`
5. `python train.py --data data/processed_2tag.csv --epochs 7 --batch-size 8 --out model_2tag_v3`
6. Swap the new models into `models\` (old ones kept as backups), then `docker compose build` and `docker compose up`.
