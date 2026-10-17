import argparse, json, sys
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

def replace_tagged(sentence,replacement):
    word=locate_tagged(sentence)
    return sentence.replace(TAG+word,replacement,1)

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description='Tagged Bangla homograph resolver: prints the output sentence only')
    p.add_argument('--sentence',type=str,required=True)
    p.add_argument('--model',default=cfg['model_dir'])
    p.add_argument('--lexicon',default=cfg['lexicon'])
    a=p.parse_args()
    if TAG not in a.sentence:print(a.sentence)
    else:
        r=Predictor(a.model,a.lexicon).predict(a.sentence)[0]
        print(replace_tagged(a.sentence,r['bangla_phoneme']))
