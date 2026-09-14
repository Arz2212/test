# ============================================================
#  Образ окружения: CSDP-SNN hyperopt
#  (Ray Tune + HyperOpt + SpikingJelly + PyTorch CPU)
#  для HPC4 (Apptainer 1.2, CentOS 7).
#
#  Сборка (из корня проекта /home/arz/SNN):
#      docker build -t <dockerhub_user>/csdp-snn:latest .
#
#  Пуш в Docker Hub:
#      docker login
#      docker push <dockerhub_user>/csdp-snn:latest
# ============================================================
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    LC_ALL=C.UTF-8

WORKDIR /app

# libgomp1 — иногда нужна torch (OpenMP-рантайм)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ВАЖНО: сначала torch/torchvision ИЗ CPU-ИНДЕКСА,
# иначе spikingjelly подтянет CUDA-версию torch (~2 ГБ, и она не нужна).
RUN pip install --upgrade pip \
    && pip install torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
        spikingjelly==0.0.0.0.14 \
        "ray[tune]==2.52.1" \
        hyperopt==0.2.7 \
        numpy==2.2.6

# Код задачи зашиваем в образ ПОСЛЕ pip: при изменении main.py
# пересобирается только этот слой (pip-слой берётся из кэша).
# Данные MNIST монтируем снаружи через --bind.
COPY spikingjell/main.py /app/main.py

# ENTRYPOINT => работает и в docker, и в apptainer:
#   docker run image --samples 1000
#   apptainer run image.sif --samples 1000
ENTRYPOINT ["python", "/app/main.py"]
