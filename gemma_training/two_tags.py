"""At most 2 <hg> tags per sentence.

The model is trained on sentences with 0, 1 or 2 tags. When a sentence has 3 or more, the first two tagged
words are corrected and everything from the third tag onward comes back exactly as it was typed, only without
the <hg> tags (the word after a 3rd or later tag is NOT respelled).
This is done here in code, so it holds whatever the model does. Used by web_app.py.
"""
import os

MAX_TAGS = int(os.environ.get("HG_MAX_TAGS", "2"))   # the models were trained with at most 2


def split_at_extra_tag(sentence, tag="<hg>", max_tags=MAX_TAGS):
    """-> (head, tail). tail starts at the word holding the 3rd tag; '' when there are 2 tags or fewer."""
    pos = -1
    for _ in range(max_tags + 1):
        pos = sentence.find(tag, pos + 1)
        if pos == -1:
            return sentence, ""
    start = sentence.rfind(" ", 0, pos) + 1          # start of the word that holds the 3rd tag
    return sentence[:start].rstrip(), sentence[start:]


def rewrite_max_two(sentence, rewrite, tag="<hg>"):
    """rewrite(s) is the model call for a sentence with at most 2 tags.
    From the 3rd tag on, the text is kept as typed with the tags removed."""
    head, tail = split_at_extra_tag(sentence, tag)
    if not tail:
        return rewrite(sentence)
    tail = " ".join(tail.replace(tag, "").split())
    # the model sees the whole sentence (more context) with the extra tags removed ...
    clean = head + " " + tail
    out = rewrite(clean).split()
    n_head = len(head.split())
    # ... and its words are kept only up to the 3rd tag; the rest is the input as typed
    if len(out) == len(clean.split()):
        return " ".join(out[:n_head] + tail.split())
    # the model added or dropped a word, so the split point is unknown: correct the first part on its own
    return " ".join((rewrite(head) + " " + tail).split())


# ---- sahajBERT: it classifies ONE tagged word per sentence, so a 2-tag sentence is sent as 2 one-tag copies ----
import re as _re

_WORD = _re.compile(r"[ঀ-৿]+")


def tagged_words(sentence, tag="<hg>"):
    """the word right after each tag, in order ('' when a tag is not followed by a Bangla word)"""
    return [(_WORD.match(p).group(0) if _WORD.match(p) else "") for p in sentence.split(tag)[1:]]


def one_tag_copies(sentence, tag="<hg>"):
    """'a <hg>x b <hg>y' -> ['a <hg>x b y', 'a x b <hg>y']: one copy per tag, the other words untagged"""
    parts = sentence.split(tag)
    return ["".join(p + (tag if j == i else "") for j, p in enumerate(parts[:-1])) + parts[-1]
            for i in range(len(parts) - 1)]


def put_respellings(sentence, respellings, tag="<hg>"):
    """replace the word after each tag, in order, by its respelling, and drop those tags"""
    parts = sentence.split(tag)
    out = [parts[0]]
    for p, new in zip(parts[1:], respellings):
        m = _WORD.match(p)
        out.append(new + p[m.end():] if m else p)
    return "".join(out)


# ---- keyword safety net for the আসর family (আসর, আসরে, আসরের) ----
# reading 1 = gathering (আশর...), reading 2 = the Asr prayer (spelling unchanged).
# The models lean on the word just before the tag, so a sentence that STARTS with আসর often comes out wrong.
# Words near the tagged word decide when only one side's cue words are present; otherwise the model decides.
ASR_FAMILY = {"আসর", "আসরে", "আসরের"}
PRAYER_CUES = ("নামাজ", "নামায", "সালাত", "ওয়াক্ত", "আজান", "আযান", "জামাত", "মসজিদ", "অজু", "ওজু",
               "মুসল্লি", "ইমাম", "জায়নামাজ", "মুয়াজ্জিন", "রাকাত", "কাজা")
PRAYER_NEXT = ("পর", "পরে", "পরপর", "পরপরই", "আগে", "আগেই", "সময়", "সময়")   # আসরের পর / আসরের আগে / আসরের সময়
GATHERING_CUES = ("গান", "গজল", "কবিতা", "আবৃত্তি", "আড্ডা", "জমে", "জমাতে", "জমিয়ে", "জমজমাট", "জমেছ", "মঞ্চ",
                  "শিল্পী", "বসেছিল", "বসেছে", "বসবে", "বসল", "বসানো", "ভাঙল", "ভাঙতে", "ভেঙে", "কবিগান", "কবিয়াল",
                  "জলসা", "মেলার", "মেলায়", "গল্প", "নাচ", "বাউল", "সাহিত্য", "তবলা", "সংগীত", "কাওয়ালি", "দর্শক",
                  "শ্রোতা", "পুঁথি", "উপস্থাপক", "মাতিয়ে", "মাতানো", "হারমোনিয়াম")
CUE_WINDOW = int(os.environ.get("HG_ASR_WINDOW", "4"))   # words on each side, within the same sentence
ASR_RULE_ON = os.environ.get("HG_ASR_RULE", "1") != "0"   # HG_ASR_RULE=0 switches the safety net off


def asr_rule(sentence, tag="<hg>"):
    """For a sentence with ONE tag on a আসর-family word -> ("1" or "2", cue word); (None, '') if the rule
    does not apply (other word, no cue, or cues of both kinds)."""
    parts = sentence.split(tag)
    if not ASR_RULE_ON or len(parts) != 2:
        return None, ""
    m = _WORD.match(parts[1])
    if not m or m.group(0) not in ASR_FAMILY:
        return None, ""
    left = _re.split(r"[।?!]", parts[0])[-1].split()[-CUE_WINDOW:]
    right = _re.split(r"[।?!]", parts[1][m.end():])[0].split()[:CUE_WINDOW]
    words = [w.strip("।,;:!?\"'‘’“”()-–—.…") for w in left + right]
    prayer = [w for w in words if w.startswith(PRAYER_CUES)]
    nxt = right[0].strip("।,;:!?\"'‘’“”()-–—.…") if right else ""
    if nxt in PRAYER_NEXT or nxt.startswith(("সময়", "সময়")):
        prayer.append(nxt)
    gathering = [w for w in words if w.startswith(GATHERING_CUES)]
    if prayer and not gathering:
        return "2", prayer[0]
    if gathering and not prayer:
        return "1", gathering[0]
    return None, ""
