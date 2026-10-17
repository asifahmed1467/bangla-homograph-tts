r"""Run a list of example sentences through the same path as app.py and print PASS / FAIL.

Run:  python test_examples.py                        (real model, adapter folder gemma_cpu_lora)
      python test_examples.py --adapter D:\path\to\gemma_cpu_lora
      python test_examples.py --sentence "আমি কেন যাব?"   (one sentence of your own, no PASS/FAIL; with no listed word plain Gemma answers it)
      python test_examples.py --lookup-only          (no model: only checks the word lookup and tagging)
      python test_examples.py --fake                 (stand-in instead of Gemma, to check this script itself)

The 22-example table checks only the rewriting path (plain-Gemma chat is not used there, so sentences without a
listed word must come back unchanged). Plain-Gemma chat is used by app.py and by --sentence.

Expected answers follow the training CSV (my_words_8.csv): the homograph word is replaced by its
bangla_phoneme spelling for that reading (e.g. কেন "why" -> ক্যানো, কেন "buys" -> কেনো); readings that are
already spelled the same stay as they are; sentences without a listed word come back unchanged.
Rows marked "csv" are taken from the CSV (the sentence with the word in its normal spelling).
Rows marked "guess" are NEW sentences: the expected answer is the author's best guess, not verified.
A FAIL on a "guess" row can be a wrong expectation, not a wrong model.
"""
import argparse, sys
from homograph_lookup import WORDS_CSV, load_words, fix_sentence, get_models

# (source, sentence, expected output, should the model be called?)
EXAMPLES = [
    # --- from the CSV: one per word and reading ---
    ("csv", "ভিডিও কলে আজ গল্পের আসর বেশ জমে উঠেছে।", "ভিডিও কলে আজ গল্পের আশর বেশ জমে উঠেছে।", True),
    ("csv", "আসরের নামাজের আগে আপনার অর্ডারের অবস্থা জানিয়ে দেব।", "আসরের নামাজের আগে আপনার অর্ডারের অবস্থা জানিয়ে দেব।", True),
    ("csv", "অনলাইন কবিতার আসরে আজ অনেক নতুন মুখ দেখা গেল।", "অনলাইন কবিতার আশরে আজ অনেক নতুন মুখ দেখা গেল।", True),
    ("csv", "অনলাইনে কেনা ফোনটি আসল কিনা বুঝতে পারছি না।", "অনলাইনে কেনা ফোনটি আশল কিনা বুঝতে পারছি না।", True),
    ("csv", "অবশেষে ডেলিভারির লোকটি আসল, তাই চিন্তা কমল।", "অবশেষে ডেলিভারির লোকটি আশলো, তাই চিন্তা কমল।", True),
    ("csv", "অর্ডারের ঠিকানাটা আবার ভালো করে যাচাই কর।", "অর্ডারের ঠিকানাটা আবার ভালো করে যাচাই করো।", True),
    ("csv", "অনলাইন কেনাকাটার রসিদে কর আলাদা করে দেখানো হয়েছে।", "অনলাইন কেনাকাটার রসিদে কর আলাদা করে দেখানো হয়েছে।", True),
    ("csv", "শিক্ষক প্রতিদিন আমাদের হাতের লেখার অনুশীলন করাত।", "শিক্ষক প্রতিদিন আমাদের হাতের লেখার অনুশীলন করাতো।", True),
    ("csv", "আমার পেমেন্ট এখনও নিশ্চিত হচ্ছে না কেন?", "আমার পেমেন্ট এখনও নিশ্চিত হচ্ছে না ক্যানো?", True),
    ("csv", "বাজার থেকে তাজা সবজি কেন।", "বাজার থেকে তাজা সবজি কেনো।", True),
    ("csv", "কেবলা নির্ধারণের অ্যাপটি খুলতে পারছি না, সাহায্য করুন।", "কেবলা নির্ধারণের অ্যাপটি খুলতে পারছি না, সাহায্য করুন।", True),
    ("csv", "কেবলা ছেলেটা সব কথায় শুধু হাসে।", "ক্যাবলা ছেলেটা সব কথায় শুধু হাসে।", True),
    # --- new sentences, expected answers are guesses ---
    ("guess", "আজকের গানের আসরে অনেক মানুষ এসেছে।", "আজকের গানের আশরে অনেক মানুষ এসেছে।", True),
    ("guess", "কেন আমার অর্ডারটা এখনো আসেনি?", "ক্যানো আমার অর্ডারটা এখনো আসেনি?", True),
    ("guess", "“কেন?” সে জিজ্ঞেস করল।", "“ক্যানো?” সে জিজ্ঞেস করল।", True),
    ("guess", "কেন সে গানের আসরে গেল না?", "ক্যানো সে গানের আশরে গেল না?", True),
    ("guess", "মা প্রতি সপ্তাহে বাজার থেকে মাছ কেন।", "মা প্রতি সপ্তাহে বাজার থেকে মাছ কেনো।", True),
    # --- no listed word: unchanged, model must not be called ---
    ("guess", "আমি প্রতিদিন সকালে এক কাপ চা খাই।", "আমি প্রতিদিন সকালে এক কাপ চা খাই।", False),
    ("guess", "বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।", "বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।", False),
    ("guess", "আজ কেনাকাটা শেষ করে করলাম রান্না, আসলে খুব ক্লান্ত।", "আজ কেনাকাটা শেষ করে করলাম রান্না, আসলে খুব ক্লান্ত।", False),  # words that only CONTAIN a listed word
    ("guess", "", "", False),
    ("guess", "This sentence is English only.", "This sentence is English only.", False),
]

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="gemma_cpu_lora", help="folder made by train_cpu.py")
ap.add_argument("--words", default=str(WORDS_CSV))
ap.add_argument("--sentence", help="run just this one sentence and print the result")
ap.add_argument("--lookup-only", action="store_true", help="no model: check only which sentences would reach the model")
ap.add_argument("--fake", action="store_true", help="stand-in instead of Gemma (checks this script, not the model)")
a = ap.parse_args()

words, said = load_words(a.words)
generate, chat = ((lambda t: t), None) if a.lookup_only else get_models(a.adapter, a.fake, said)
if a.fake:
    print("NOTE: --fake is a stand-in, not Gemma. These results say nothing about the trained model.\n")

if a.sentence:
    r = fix_sentence(a.sentence, words, generate, chat)
    print("input  :", r["input"])
    print("tagged :", r["tagged"])
    print("path   :", {"rewrite": "homograph found (" + ", ".join(r["found"]) + ") -> fine-tuned adapter",
                       "chat": "no listed word -> adapter off, plain Gemma chat",
                       "unchanged": "no listed word, nothing to do"}[r["mode"]])
    print("output :", r["output"])
    sys.exit(0)

stats = {"csv": [0, 0], "guess": [0, 0]}
bad_call = 0
for i, (src, sentence, expected, want_call) in enumerate(EXAMPLES, 1):
    r = fix_sentence(sentence, words, generate)
    call_ok = r["called_model"] == want_call
    ok = call_ok if a.lookup_only else (r["output"] == expected and call_ok)
    stats[src][0] += ok
    stats[src][1] += 1
    bad_call += not call_ok
    print(f"[{'PASS' if ok else 'FAIL'}] #{i} ({src}) model called: {'yes' if r['called_model'] else 'no'}"
          f"{'' if call_ok else '  <-- expected ' + ('yes' if want_call else 'no')}")
    if not ok:
        print(f"   input   : {r['input']}")
        print(f"   tagged  : {r['tagged']}")
        if not a.lookup_only:
            print(f"   expected: {expected}")
            print(f"   got     : {r['output']}")

total = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
print(f"\nSUMMARY: {total[0]}/{total[1]} passed"
      f"  (csv sentences {stats['csv'][0]}/{stats['csv'][1]}, new guessed sentences {stats['guess'][0]}/{stats['guess'][1]})")
print(f"model called when it should / not called when it should not: {len(EXAMPLES) - bad_call}/{len(EXAMPLES)}")
if a.lookup_only:
    print("(lookup-only: the model was not run, output text was not checked)")
sys.exit(0 if total[0] == total[1] else 1)
