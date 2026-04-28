#!/bin/bash
#SBATCH --job-name=PretrainPlots
#SBATCH --output=logs/Logs_Plots_%j.out
#SBATCH --error=logs/Logs_Plots_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=dev_gpu_h100

set -euo pipefail

echo "Job started ..."

source ~/.bashrc
conda activate llm-env

cd /pfs/data6/home/ul/ul_student/ul_raf24/project/Masterarbeit

export PYTHONUNBUFFERED=1
mkdir -p logs
mkdir -p my_gpt2/results/plots

python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"

python my_gpt2/results/plots/generate_pretraining_plots.py \
  --input-root my_gpt2/results/pretraining \
  --output-dir my_gpt2/results/plots \
  --formats png \
  --dpi 300

echo "Job completed."
