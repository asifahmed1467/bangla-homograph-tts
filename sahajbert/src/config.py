"""Load config.json; env var HG_<KEY> overrides each key. CLI flags still win (handled in scripts)."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config.json'
PATH_KEYS = ('raw_data', 'processed_data', 'lexicon', 'model_dir')

def env_name(key):
    return 'HG_' + key.upper()

def load(verbose=True):
    cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    from_env = {}
    for key, default in list(cfg.items()):
        raw = os.environ.get(env_name(key))
        if raw is None or raw == '':
            continue
        try:
            cfg[key] = type(default)(raw) if not isinstance(default, str) else raw
        except ValueError:
            sys.exit(f'Bad value for {env_name(key)}={raw!r}: expected {type(default).__name__}')
        from_env[key] = cfg[key]
    if verbose:
        # stderr, so scripts that print only a result on stdout stay clean
        for key, value in from_env.items():
            print(f'[config] {key} = {value!r} (from env {env_name(key)})', file=sys.stderr)
    for key in PATH_KEYS:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    cfg['_from_env'] = from_env
    return cfg
