import csv, json, re
from pathlib import Path

TAG = '<hg>'
COLS = ['homograph','phoneme','sentence','pronunciation_variant','bangla_phoneme']

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def normalize_training_row(row):
    word = row['homograph'].strip()
    sentence = row['sentence'].strip().replace(TAG, '')
    spoken = row['bangla_phoneme'].strip()
    # Match whole word, not inside another Bengali word.
    boundary = r'(?<![\u0980-\u09ff]){}(?![\u0980-\u09ff])'
    for form in sorted(set([spoken,word]), key=len, reverse=True):
        matches=list(re.finditer(boundary.format(re.escape(form)), sentence))
        if len(matches) == 1:
            m=matches[0]
            canonical = sentence[:m.start()] + TAG + word + sentence[m.end():]
            return canonical
    raise ValueError(f'Expected exactly one target for {word!r}: {sentence!r}')

def locate_tagged(sentence):
    if sentence.count(TAG) != 1:
        raise ValueError('Each sentence must contain exactly one <hg> marker immediately before the target word.')
    tail = sentence.split(TAG,1)[1]
    match = re.match(r'[\u0980-\u09ff]+',tail)
    if not match: raise ValueError('<hg> must be immediately before a Bengali word; no space is allowed.')
    return match.group(0)
