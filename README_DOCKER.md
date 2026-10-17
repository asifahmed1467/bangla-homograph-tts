# Run the demo with Docker (Windows, CPU only)

One web page, three choices in a dropdown:

| Dropdown | What it does |
|---|---|
| Gemma 3 270M | One option, decided per sentence: a sentence **with** `<hg>` is converted by the homograph adapter (tag removed, word replaced by its respelling); a sentence **without** `<hg>` gets plain Gemma (adapter off) and its normal answer. |
| sahajBERT homograph classifier | Sentences with `<hg>` are converted (all lines in one call); lines without `<hg>` come back unchanged. |

Type one sentence, or many (one per line). Example: `তুমি <hg>কেন এত দেরি করলে?`

Nothing is retrained.

## Everything is inside the image

`docker build` copies your trained models from `models/` and downloads the Gemma base model into the image. After that, running it needs no download, no volume and no internet. Trade-off: the image is a bit over 1 GB, and you must rebuild it if you change the models. If a trained model is missing from `models/`, the build stops and says which one.

The weights are big, so `models/` is **not in git**; the repo holds only the Dockerfile.

## Before you build: fill `models/`

```
models/
  gemma_cpu_lora/   <- your whole gemma_cpu_lora folder
  sahajbert/        <- the contents of sahajbert/model (config.json, model.safetensors, tokenizer files)
```

## Steps (one at a time)

1. Install Docker Desktop for Windows, start it, wait for "Engine running". In PowerShell: `docker --version`
2. Open PowerShell in this folder and check: `docker compose version`
3. Put your models in `models/` (above).
4. Build once (internet needed, several minutes): `docker compose build`
5. Start: `docker compose up`, then open http://localhost:7860 when the log says so.
6. Stop with `Ctrl+C`. Next time only `docker compose up`.

The page expects a Gemma adapter trained with `--tag_mode before` (the default now). For an older adapter trained with `run.py`'s "trigger" mode, set `HG_TAG_MODE=trigger`.

## Sharing with supervisors

Since the models are baked in, share the built image, not the repo alone. On your PC, after `docker compose build`:

```
docker save bangla-homograph:latest -o bangla-homograph.tar
```

Send the .tar (about 1-2 GB, e.g. via Drive/USB). On their PC with Docker Desktop:

```
docker load -i bangla-homograph.tar
docker run -p 7860:7860 bangla-homograph:latest
```

then open http://localhost:7860. They need no internet and no models folder. (Pushing to Docker Hub/GHCR works too, only if you ask for it; trained weights would then be public unless the repo is private.)
