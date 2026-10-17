Put your trained models here BEFORE building the image. They are baked into the Docker image.
This folder is kept out of git (weights are big), so only the Dockerfile lives in the repo.

models/
  gemma_cpu_lora/      <- the whole folder made by gemma_training/train_cpu.py
                          (adapter_config.json, adapter_model.safetensors, base_model.txt, tokenizer files)
  sahajbert/           <- the CONTENTS of sahajbert/model made by sahajbert/train.py
                          (config.json, model.safetensors, tokenizer files)

If either is missing, `docker compose build` stops with a message saying which one.
