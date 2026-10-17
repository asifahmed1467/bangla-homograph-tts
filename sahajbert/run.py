import os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config

def script(name, *args):
    return subprocess.call([sys.executable, str(ROOT / name), *args], cwd=ROOT)

def show_config():
    cfg = config.load(verbose=False)
    print(f'Config file: {config.CONFIG_PATH}')
    for key, value in cfg.items():
        if key.startswith('_'):
            continue
        src = f'env {config.env_name(key)}' if key in cfg['_from_env'] else 'config.json'
        print(f'  {key:15} = {value!r:40} [{src}]')
    if not cfg['_from_env']:
        print('No HG_* environment variables are set.')

def type_sentence():
    print('Type a sentence with <hg> right before the target word (no space), e.g. তুমি <hg>কেন এত দেরি করলে?')
    s = input('Sentence: ').strip()
    if not s:
        print('Nothing entered.')
        return
    script('predict_sentence.py', '--sentence', s)

ACTIONS = {'1': lambda: script('prepare.py'), '2': lambda: script('train.py'),
           '3': type_sentence, '4': show_config}

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    while True:
        print('\n1) Prepare data\n2) Train\n3) Type a tagged sentence\n4) Show config\n0) Quit')
        try:
            choice = input('Choose: ').strip()
        except EOFError:
            break
        if choice == '0':
            break
        action = ACTIONS.get(choice)
        if action:
            action()
        else:
            print('Please type 0, 1, 2, 3 or 4.')

if __name__ == '__main__':
    main()
