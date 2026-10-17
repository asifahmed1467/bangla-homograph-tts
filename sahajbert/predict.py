import argparse, json, csv
from pathlib import Path
import torch
from transformers import AlbertForSequenceClassification, PreTrainedTokenizerFast
from src.common import TAG,locate_tagged
from src import config

cfg=config.load()

class Predictor:
    def __init__(self,model_dir=None,lexicon_path=None):
        model_dir=model_dir or cfg['model_dir'];lexicon_path=lexicon_path or cfg['lexicon']
        self.lexicon=json.loads(Path(lexicon_path).read_text(encoding='utf-8'))
        self.tok=PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.model=AlbertForSequenceClassification.from_pretrained(model_dir).to('cpu').eval()
        torch.set_num_threads(min(cfg['threads'],torch.get_num_threads()))
    def predict(self,sentences,batch_size=None):
        batch_size=batch_size or cfg['batch_size']
        if isinstance(sentences,str):sentences=[sentences]
        if not isinstance(sentences,list):raise TypeError('Expected a string or list of strings')
        words=[]
        for sentence in sentences:
            word=locate_tagged(sentence)
            if word not in self.lexicon:raise ValueError(f'Unknown homograph: {word!r}; available: {list(self.lexicon)}')
            words.append(word)
        results=[]
        for start in range(0,len(sentences),batch_size):
            batch=sentences[start:start+batch_size]
            x=self.tok(batch,return_tensors='pt',padding=True,truncation=True,max_length=cfg['max_length'])
            with torch.inference_mode():
                probs=torch.softmax(self.model(**x).logits,dim=-1)
            for i,(sentence,prob) in enumerate(zip(batch,probs)):
                word=words[start+i]
                variant=int(prob.argmax().item())+1
                mapping=self.lexicon[word][str(variant)]
                results.append({'sentence':sentence,'homograph':word,'pronunciation_variant':variant,'phoneme':mapping['phoneme'],'bangla_phoneme':mapping['bangla_phoneme'],'confidence':round(prob.max().item(),4)})
        return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Tagged Bangla homograph batch predictor')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--sentence',type=str)
    group.add_argument('--input',type=str,help='JSON array file of tagged Bangla sentences')
    p.add_argument('--model',default=cfg['model_dir'])
    p.add_argument('--lexicon',default=cfg['lexicon'])
    p.add_argument('--batch-size',type=int,default=cfg['batch_size'])
    p.add_argument('--output',default='',help='Optional UTF-8 .json or .csv result path')
    a=p.parse_args()
    data=[a.sentence] if a.sentence is not None else json.loads(Path(a.input).read_text(encoding='utf-8-sig'))
    results=Predictor(a.model,a.lexicon).predict(data,a.batch_size)
    if a.output:
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        if output.suffix.lower()=='.csv':
            with output.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(results[0]) if results else ['sentence','homograph','pronunciation_variant','phoneme','bangla_phoneme','confidence'])
                w.writeheader();w.writerows(results)
        else:output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Saved output to:',output)
    else:print(json.dumps(results,ensure_ascii=False,indent=2))
