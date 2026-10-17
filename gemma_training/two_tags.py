"""At most 2 <hg> tags per sentence.

The model is trained on sentences with 0, 1 or 2 tags. When a sentence has 3 or more, the first two tagged
words are corrected and everything from the third tag onward comes back exactly as it was typed, only without
the <hg> tags (the word after a 3rd or later tag is NOT respelled).
This is done here in code, so it holds whatever the model does. Used by web_app.py, evaluate_cpu.py, run_local.py.
"""
MAX_TAGS = 2


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
