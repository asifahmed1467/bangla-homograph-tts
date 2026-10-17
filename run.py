"""One menu for both models. Pick the model in config.json ("model": "gemma" or "sahajbert"),
or override it with the environment variable HG_MODEL. Menu option 1 changes config.json for you."""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'config.json'
MODELS = {
    'gemma': {
        'dir': 'gemma_training',
        'prepare': ['prepare_data.py', '--smoke', '--csv', 'my_words_8.csv', '--test_frac', '0.2',
                    '--tag', '<hg>', '--tag_mode', 'trigger'],
        'train': ['train_cpu.py', '--epochs', '3'],
        'predict': ['evaluate_cpu.py', '--adapter', 'gemma_cpu_lora', '--sentence'],
        'hint': 'Type a normal sentence, e.g. আমার পেমেন্ট এখনও নিশ্চিত হচ্ছে না কেন?',
    },
    'sahajbert': {
        'dir': 'sahajbert',
        'prepare': ['prepare.py'],
        'train': ['train.py'],
        'predict': ['predict_sentence.py', '--sentence'],
        'hint': 'Put <hg> right before the word (no space), e.g. তুমি <hg>কেন এত দেরি করলে?',
    },
}


def read_config():
    return json.loads(CONFIG.read_text(encoding='utf-8-sig'))


def current_model():
    """HG_MODEL (environment) wins over config.json."""
    env = os.environ.get('HG_MODEL', '').strip().lower()
    name = env or str(read_config().get('model', 'gemma')).strip().lower()
    if name not in MODELS:
        sys.exit(f'Unknown model {name!r}. Use one of: {", ".join(MODELS)} '
                 f'(set in config.json or with HG_MODEL).')
    return name, ('environment HG_MODEL' if env else 'config.json')


def run_script(model, *args):
    folder = ROOT / MODELS[model]['dir']
    return subprocess.call([sys.executable, *args], cwd=folder)


def choose_model():
    names = list(MODELS)
    for i, n in enumerate(names, 1):
        print(f'  {i}) {n}')
    pick = input('Which model? ').strip()
    if pick.isdigit() and 1 <= int(pick) <= len(names):
        cfg = read_config()
        cfg['model'] = names[int(pick) - 1]
        CONFIG.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
        print(f'Saved: model = {cfg["model"]} in config.json')
        if os.environ.get('HG_MODEL'):
            print('Note: HG_MODEL is set in this window and still wins. Clear it with: set HG_MODEL=')
    else:
        print('No change.')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    while True:
        model, source = current_model()
        m = MODELS[model]
        print(f'\nCurrent model: {model}  (from {source})')
        print('1) Choose model\n2) Prepare data\n3) Train\n4) Try one sentence\n0) Quit')
        try:
            c = input('Choose: ').strip()
        except EOFError:
            break
        if c == '0':
            break
        elif c == '1':
            choose_model()
        elif c == '2':
            run_script(model, *m['prepare'])
        elif c == '3':
            run_script(model, *m['train'])
        elif c == '4':
            print(m['hint'])
            s = input('Sentence: ').strip()
            if s:
                run_script(model, *m['predict'], s)
        else:
            print('Please type 0, 1, 2, 3 or 4.')


if __name__ == '__main__':
    main()
