#!/bin/bash
# ============================================================
#  Запуск гиперопт-поиска CSDP-SNN на HPC4 (Apptainer)
#  Запуск:  sbatch job.slurm
# ============================================================
#SBATCH --job-name=csdp_search
#SBATCH --partition=PUT_YOUR_CPU_QUEUE_HERE   # ваша CPU-очередь, напр. ...-el7-...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=csdp_search_%j.log

# Ray должен видеть ядра, выделенные SLURM (а не все ядра узла)
export RAY_NUM_CPUS=$SLURM_CPUS_PER_TASK
# Временные файлы Ray — в TMPDIR задания, если он задан
export RAY_TMPDIR=${TMPDIR:-/tmp}

# Данные MNIST должны лежать на кластере, напр. в $HOME/mnist/.
# Bind-mount подставляет их на жёстко прописанный в main.py путь:
#   /home/arz/SNN/jax_version/cSNN/data/mnist/mnist
apptainer run \
    --bind "$HOME/mnist:/home/arz/SNN/jax_version/cSNN/data/mnist/mnist" \
    "$HOME/containers/csdp_snn.sif" \
    --samples 1000
