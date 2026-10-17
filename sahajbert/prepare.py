import argparse, collections, csv, random
from pathlib import Path
from src.common import read_csv, normalize_training_row, save_json, COLS
from src import config

cfg=config.load()

p=argparse.ArgumentParser()
p.add_argument('--input',default=cfg['raw_data'])
p.add_argument('--output',default=cfg['processed_data'])
p.add_argument('--val-ratio',type=float,default=cfg['val_ratio'])
p.add_argument('--seed',type=int,default=cfg['seed'])
a=p.parse_args()
rows=read_csv(a.input)
for c in COLS:
    if c not in (rows[0] if rows else {}): raise ValueError(f'Missing CSV column: {c}')
clean=[]; lexicon={}; seen=set(); invalid=[]
for i,r in enumerate(rows,2):
    try:
        word=r['homograph'].strip(); variant=str(r['pronunciation_variant']).strip()
        if variant not in ('1','2'): raise ValueError('variant must be 1 or 2')
        tagged=normalize_training_row(r)
        key=(word,tagged)
        if key in seen:
            invalid.append((i,'duplicate sentence',tagged));continue
        seen.add(key)
        entry={'phoneme':r['phoneme'].strip(),'bangla_phoneme':r['bangla_phoneme'].strip()}
        if word in lexicon and variant in lexicon[word] and lexicon[word][variant]!=entry:
            raise ValueError('conflicting phoneme mappings')
        lexicon.setdefault(word,{})[variant]=entry
        clean.append({'homograph':word,'tagged_sentence':tagged,'pronunciation_variant':int(variant),**entry})
    except Exception as ex:
        invalid.append((i,str(ex),r.get('sentence','')))
if not clean:raise RuntimeError('No valid data')
if not (0<a.val_ratio<1):raise ValueError('val-ratio must be between 0 and 1')
# Group by contextual skeleton to keep identical normalized sentences together.
# This does NOT guarantee near-paraphrases are disjoint.
groups=collections.defaultdict(list)
for r in clean:groups[(r['homograph'],r['tagged_sentence'])].append(r)
rng=random.Random(a.seed)
by_class=collections.defaultdict(list)
for key,records in groups.items():by_class[(records[0]['homograph'],records[0]['pronunciation_variant'])].append(records)
train=[];val=[]
for _,chunks in sorted(by_class.items()):
    rng.shuffle(chunks)
    n_val=max(1,round(len(chunks)*a.val_ratio)) if len(chunks)>1 else 0
    for i,chunk in enumerate(chunks): (val if i<n_val else train).extend(chunk)
rng.shuffle(train);rng.shuffle(val)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['split','homograph','tagged_sentence','pronunciation_variant','phoneme','bangla_phoneme'])
    w.writeheader();w.writerows([{'split':'train',**r} for r in train]+[{'split':'val',**r} for r in val])
save_json(cfg['lexicon'],lexicon)
print(f'Input {len(rows)}, train {len(train)}, validation {len(val)}, excluded {len(invalid)}, homographs {len(lexicon)}')
for issue in invalid[:25]:print('WARNING row',*issue,sep=' | ')
print('Saved:',out,'and',cfg['lexicon'])
print('NOTE: review semantic labels and near-paraphrase leakage before production training.')
