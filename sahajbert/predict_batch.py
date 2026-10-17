"""Batch resolver: tagged sentences in, untagged sentences with the predicted pronunciation out.

  python predict_batch.py --input batch_test.json [--output out.json]

Model is built once; all tagged sentences go through Predictor.predict in one call.
Sentences without <hg> are returned unchanged and never sent to the model.
"""
import argparse, json, sys
from pathlib import Path
from src.common import TAG, locate_tagged
from src import config

cfg = config.load()  # env override messages go to stderr

def replace_tagged(sentence, replacement):
    # same logic as predict_sentence.replace_tagged (duplicated so importing this
    # module does not need torch until a Predictor is actually built)
    word = locate_tagged(sentence)
    return sentence.replace(TAG + word, replacement, 1)

def fix_sentences(sentences, predictor=None, model=None, lexicon=None, batch_size=None):
    """list[str] -> list[str], same order. Builds a Predictor only if needed and none is given."""
    sentences = list(sentences)
    idx = [i for i, s in enumerate(sentences) if TAG in s]
    out = list(sentences)
    if not idx:
        return out
    if predictor is None:
        from predict_sentence import Predictor
        predictor = Predictor(model or cfg['model_dir'], lexicon or cfg['lexicon'])
    results = predictor.predict([sentences[i] for i in idx], batch_size or cfg['batch_size'])
    if len(results) != len(idx):
        raise RuntimeError('Predictor returned %d results for %d sentences' % (len(results), len(idx)))
    for i, r in zip(idx, results):
        out[i] = replace_tagged(sentences[i], r['bangla_phoneme'])
    return out

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='Tagged Bangla homograph batch resolver')
    p.add_argument('--input', required=True, help='JSON array of tagged sentences (utf-8 / utf-8-sig)')
    p.add_argument('--output', default='', help='Optional JSON file: [{"input":..,"output":..}]')
    p.add_argument('--model', default=cfg['model_dir'])
    p.add_argument('--lexicon', default=cfg['lexicon'])
    p.add_argument('--batch-size', type=int, default=cfg['batch_size'])
    a = p.parse_args()
    data = json.loads(Path(a.input).read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        sys.exit('--input must be a JSON array of strings')
    outs = fix_sentences(data, model=a.model, lexicon=a.lexicon, batch_size=a.batch_size)
    for s, o in zip(data, outs):
        print('input : ' + s)
        print('output: ' + o)
        print()
    if a.output:
        path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{'input': s, 'output': o} for s, o in zip(data, outs)], ensure_ascii=False, indent=2), encoding='utf-8')
        print('Saved output to:', path, file=sys.stderr)

if __name__ == '__main__':
    main()
