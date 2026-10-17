# Bangla homograph demo: Gemma 3 270M (+ LoRA adapter) and sahajBERT, CPU only.
# Everything is baked into the image: your trained models (from ./models) and the Gemma base model.
# Run time needs no download, no volume and no internet.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/hf_cache \
    HG_MODELS_DIR=/models

WORKDIR /app

# CPU-only torch (much smaller than the default GPU build)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

# your trained weights (not in git): models/gemma_cpu_lora and models/sahajbert
COPY models /models
COPY build_check_and_fetch.py .
RUN python build_check_and_fetch.py

COPY web_app.py run.py config.json ./
COPY gemma_training gemma_training
COPY sahajbert sahajbert
COPY homograph_call_context homograph_call_context
COPY tts_readiness tts_readiness

# from here on the container never goes to the internet
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["python", "web_app.py"]
