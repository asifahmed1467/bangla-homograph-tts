# How training works, explained with your homograph data

This guide follows one sentence of yours from the CSV to the finished model. Each section matches one step of the pipeline in [README.md](README.md).

```
CSV rows ──► prepare_data.py ──► train / val / test files
                                       │
                 Gemma 4 (already knows Bangla) + LoRA (new, small)
                                       │
                               train.py: show examples,
                               measure the mistake, nudge, repeat
                                       │
                               evaluate.py: score on unseen sentences
                                       │
                               use it: sentence in ──► fixed sentence out
```

---

## 1. The big idea: learning from examples

Normal programming means you write the rules yourself, for example "if the sentence has নামাজ near আসর, write আসর". That breaks quickly, because there are too many contexts.

Training works the other way round. You show the model hundreds of examples of *question → correct answer*, and it adjusts itself until its answers match yours. You never write the rule. The model finds the pattern ("prayer words nearby means আসর, gathering words mean আশর") from the examples.

That is why the data matters more than anything else:
- **Wrong labels teach wrong answers.** The model copies your CSV, mistakes included.
- **Missing situations are not learned.** If every আসর (prayer) sentence talks about online orders, the model may learn "order = prayer" instead of "নামাজ = prayer".

---

## 2. Gemma already knows a lot, and you only teach the task

Gemma 4 was already trained by Google on a huge amount of text in many languages, including Bangla. It knows Bangla words, grammar, and which words appear together. This is called **pretraining**, and it costs millions of dollars, so you never do it yourself.

What you do is **fine-tuning**: a short extra training on your own small dataset that teaches one specific job. That job is "rewrite the homograph in its pronunciation spelling, change nothing else".

Think of a person who already speaks Bangla well. You don't teach them the language. You give them a few hundred worked examples of your new spelling convention, and they pick it up.

---

## 3. Step 1: preparing the data (`prepare_data.py`)

### 3a. Turning rows into input → output pairs

Your CSV row:

| homograph | pronunciation_variant | sentence_homograph_form | sentence_tts |
|---|---|---|---|
| আসর | 1 | ভিডিও কলে আজ গল্পের **আসর** বেশ জমে উঠেছে। | ভিডিও কলে আজ গল্পের **আশর** বেশ জমে উঠেছে। |

becomes one training example:

```
input : ভিডিও কলে আজ গল্পের আসর বেশ জমে উঠেছে।     ← how people normally type it
output: ভিডিও কলে আজ গল্পের আশর বেশ জমে উঠেছে।     ← what the TTS should read
```

### 3b. Adding plain sentences (output = input)

If the model only ever sees sentences where something changes, it learns "always change something". Then it starts damaging normal sentences. So the script adds sentences that contain no homograph, where the correct answer is the same sentence:

```
input : বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।
output: বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।
```

The script checks every word against your 160-word list, so a plain sentence never contains a homograph by accident.

Some of your homograph rows already have output = input as well. For example, বাস (bus) stays বাস. These teach the model that seeing a homograph does not always mean changing it. It has to decide from context.

### 3c. Splitting into train, val and test

This is the most important habit in machine learning. **Never test on what you trained on.** A student who memorised the answer sheet scores 100% on that same sheet, but that tells you nothing.

For every word and every reading, the script splits the sentences into three groups:

| File | Share | Used for |
|---|---|---|
| `train.jsonl` | ~70-80% | The model learns from these. |
| `val.jsonl` | ~5-10% | Checked during training, to see if the model is improving or only memorising. |
| `test.jsonl` | ~15% | Used once at the end for the final score. The model never sees these during training. |

With your data that came out as 1,320 train, 190 val and 370 test homograph rows, plus the plain sentences.

---

## 4. What the model actually sees

### 4a. A prompt in chat format

Gemma is a chat model, so each example is written as a short conversation. Simplified, it looks like this:

```
<user turn>
  If this Bangla sentence has a homograph, rewrite it in its pronunciation
  spelling according to context. Change nothing else. If there is no
  homograph, return the sentence unchanged.

  ভিডিও কলে আজ গল্পের আসর বেশ জমে উঠেছে।
<model turn>
  ভিডিও কলে আজ গল্পের আশর বেশ জমে উঠেছে।
```

The instruction (in `common.py`) is the same every time. Training and testing use exactly the same wording, otherwise the model gets confused.

### 4b. Tokens: how text becomes numbers

A model cannot read letters. It only works with numbers. A **tokenizer** cuts text into small pieces called **tokens**. A token can be a whole word, part of a word, or a single character. Each token has an ID number.

```
"গল্পের আশর বেশ"  →  [গল্প][ের][ আ][শর][ বেশ]  →  [ 48211, 873, 1290, 55102, 9934 ]
                     (illustration only; the real split depends on Gemma's tokenizer)
```

`MAX_SEQ_LEN = 256` in `common.py` means each example may be at most 256 tokens long. Your sentences are short, so they fit easily.

### 4c. The model's one skill: predict the next token

At its core, Gemma does one thing. Given all the tokens so far, it guesses the next token. To write a whole answer, it guesses one token, adds it, guesses the next, and so on.

So for your task it is guessing:

```
... গল্পের   →  next token?   আ (very likely)
... গল্পের আ →  next token?   শর (70%)  or  সর (30%)   ← the homograph decision
```

Training pushes that 70/30 toward 100/0 in the right direction, based on the context words.

We only train on the **answer** part (`train_on_responses_only` in `train.py`). The model is not asked to learn to write the instruction, only to produce the correct answer.

---

## 5. Step 2: training (`train.py`)

### 5a. One training step

1. Take a small group of examples. This is called a **batch**.
2. The model predicts the answer, token by token.
3. Compare its prediction with your correct answer and compute the **loss**. Loss is one number that says how wrong it was: high means very wrong, close to 0 means nearly perfect.
4. Adjust the model's numbers (called **weights**) a tiny bit in the direction that makes the loss smaller. The **learning rate** (`2e-4`) controls how big that adjustment is. Too big makes it jump around, and too small makes it learn very slowly.
5. Repeat with the next batch.

### 5b. Batches, steps and epochs, with your numbers

- `per_device_train_batch_size=8` with `gradient_accumulation_steps=2` means the model learns from 16 examples per step. Accumulation lets a small GPU act like it has a bigger batch.
- With about 2,500 training examples (1,320 homograph + about 1,200 plain), one pass over all of them is about 160 steps. One full pass is called an **epoch**.
- `num_train_epochs=3` means it sees every example 3 times, about 480 steps in total.

### 5c. Watching loss: learning or memorising?

During training you'll see two losses printed:

| What you see | What it means | What to do |
|---|---|---|
| train loss ↓ and eval loss ↓ | It is learning the pattern. | Good, keep going. |
| train loss ↓ but eval loss ↑ | It is **memorising** the training sentences (called overfitting). | Use fewer epochs, or add more varied data. |
| both stay high | It is not learning. | Check the data, or try a slightly higher learning rate. |

`eval loss` is measured on `val.jsonl`, sentences the model is not trained on. That is why you keep a validation set.

### 5d. LoRA: training a small add-on, not the whole model

Gemma 4 E2B has billions of weights. Changing all of them needs a huge GPU and risks making it forget what it knows.

**LoRA** (Low-Rank Adaptation) freezes all the original weights. It adds small extra layers next to them, and only those learn. They are often less than 1% of the size.

- It needs much less memory, so it fits on a gaming GPU.
- The original Bangla knowledge stays intact.
- The result is a small folder (`gemma_homograph_lora/`, tens of MB) instead of a whole new model. You can keep several adapters for different tasks.

`r=16` in `train.py` is the size of those add-on layers. Bigger means it can learn more but needs more memory. 8 to 32 is normal.

### 5e. 4-bit: shrinking the model to fit

Each weight is normally stored with 16 bits of precision. `load_in_4bit=True` stores the frozen original weights in 4 bits instead, so they take about a quarter of the memory. Quality drops only a little, and the LoRA add-on still trains in full precision. LoRA on top of a 4-bit model is called **QLoRA**.

---

## 6. Step 3: checking that it worked (`evaluate.py`)

The script gives each sentence in `test.jsonl` to the trained model and compares the answer with your CSV, letter by letter (**exact match**).

You get two scores. You need both to be high:

```
homograph sentences fixed correctly: 331/370 = 89.5%   ← picks the right reading
plain sentences left unchanged:      221/225 = 98.2%   ← doesn't damage normal text
words with mistakes: পরে (60%), মেলা (70%), ...
```
*(These numbers are made up to show the format. Your real numbers come from your run.)*

- A low homograph score on one word usually means that word has too few or too similar sentences. Check `contexts_to_add` in `tts_readiness/word_status.csv`.
- A low plain-sentence score means it changes things it should not. Add more plain sentences (`--wiki 3000`).
- Open `predictions.csv` and read the mistakes yourself. You will often find that the CSV label was wrong, not the model.

Also run the score once **before** training (`--adapter unsloth/gemma-4-E2B-it`). The difference shows what training actually added.

---

## 7. Step 4: using the trained model

```bash
python evaluate.py --sentence "আসর নামাজের পর আমরা বাজারে যাব।"
→ আসর নামাজের পর আমরা বাজারে যাব।
```

In a real system it sits in front of the TTS:

```
user text ──► Gemma + your LoRA ──► text with homographs respelled ──► TTS ──► audio
```

---

## 8. Improving it: the usual loop

Training is rarely done in one go. The normal cycle is:

1. Train.
2. Evaluate and read the mistakes in `predictions.csv`.
3. Fix or add data for the weak words. Use more contexts, more sentences, and correct the wrong labels.
4. Run `prepare_data.py` again, then train again.

Most of the improvement comes from step 3, not from changing training settings.

---

## Word list

| Word | Meaning |
|---|---|
| Model | A big mathematical function with billions of adjustable numbers (weights). |
| Weights | The numbers inside the model. Training changes them. |
| Pretraining | Google's huge initial training. You don't do this. |
| Fine-tuning | Your short extra training for one task. |
| Token | A small piece of text, turned into a number. |
| Batch | The group of examples used for one learning step. |
| Step | One "predict, measure loss, adjust" round. |
| Epoch | One full pass over all training examples. |
| Loss | How wrong the model was. Lower is better. |
| Learning rate | How big each adjustment is. |
| Overfitting | Memorising the training sentences instead of learning the pattern. |
| Validation set | Unseen sentences checked during training to catch overfitting. |
| Test set | Unseen sentences used once at the end for the final score. |
| LoRA | Small trainable add-on layers; the original model stays frozen. |
| 4-bit / QLoRA | The frozen model stored in a compressed form so it fits on a smaller GPU. |
| Adapter | The saved LoRA folder, which is your trained result. |
| Inference | Using the trained model to answer new sentences. |
