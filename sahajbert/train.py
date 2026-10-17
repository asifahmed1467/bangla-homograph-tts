import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertForSequenceClassification, PreTrainedTokenizerFast
from src.common import TAG,read_csv
from src import config

cfg=config.load()

p=argparse.ArgumentParser()
p.add_argument('--data',default=cfg['processed_data'])
p.add_argument('--base',default=cfg['model_name'])
p.add_argument('--out',default=cfg['model_dir'])
p.add_argument('--epochs',type=int,default=cfg['epochs'])
p.add_argument('--batch-size',type=int,default=cfg['batch_size'])
p.add_argument('--lr',type=float,default=cfg['lr'])
p.add_argument('--max-length',type=int,default=cfg['max_length'])
p.add_argument('--seed',type=int,default=cfg['seed'])
a=p.parse_args()
random.seed(a.seed);torch.manual_seed(a.seed);np.random.seed(a.seed)
torch.set_num_threads(min(cfg['threads'],torch.get_num_threads()))
rows=read_csv(a.data)
train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
if not train or not val:raise ValueError('Training and validation sets cannot be empty; run prepare.py')

# This is a genuine tokenizer special token and the encoder embeddings are resized accordingly.
tok=PreTrainedTokenizerFast.from_pretrained(a.base)
tok.add_special_tokens({'additional_special_tokens':[TAG]})
model=AlbertForSequenceClassification.from_pretrained(a.base,num_labels=2,ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tok))
model.to('cpu')

class HGDset(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

def collate(records):
    x=tok([r['tagged_sentence'] for r in records],padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
    x['labels']=torch.tensor([int(r['pronunciation_variant'])-1 for r in records],dtype=torch.long)
    return x
trainloader=DataLoader(HGDset(train),batch_size=a.batch_size,shuffle=True,collate_fn=collate)
valloader=DataLoader(HGDset(val),batch_size=a.batch_size,collate_fn=collate)
optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
best=-1.
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for epoch in range(1,a.epochs+1):
    model.train();losses=[]
    for batch in trainloader:
        optimizer.zero_grad(set_to_none=True)
        loss=model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();losses.append(loss.item())
    model.eval();correct=total=0
    with torch.inference_mode():
        for batch in valloader:
            labels=batch.pop('labels')
            predicted=model(**batch).logits.argmax(dim=-1)
            correct+=(predicted==labels).sum().item();total+=len(labels)
    accuracy=correct/total
    print(f'Epoch {epoch}/{a.epochs} | loss={sum(losses)/len(losses):.4f} | val_accuracy={accuracy:.3f} ({correct}/{total})',flush=True)
    if accuracy>best:
        best=accuracy
        model.save_pretrained(out,safe_serialization=True)
        tok.save_pretrained(out)
print('Best validation accuracy:',round(best,4),'Saved:',out)
