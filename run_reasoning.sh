#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/logs_reasoning_true_%j.out
#SBATCH --error=logs/logs_reasoning_true_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=127G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il


echo "Job started ..."

source ~/.bashrc
conda activate llm-env

cd /pfs/data6/home/ul/ul_student/ul_raf24/project/Masterarbeit

export PYTHONUNBUFFERED=1
mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# # No Reasoning
# python -m my_gpt2.source.sft_train \
#   --data_dir my_gpt2/source/datasets/gsm8k_sft_examples/gsm8k_direct \
#   --prefix gsm8k_direct \
#   --pretrain_ckpt my_gpt2/results/sft_ultrachat/best.pt \
#   --out_dir my_gpt2/results/sft_gsm8k_direct \
#   --max_steps 1170 \
#   --eval_every 200 \
#   --max_train_examples -1 \
#   --save_every 500

# With Reasoning (COT)
python -m my_gpt2.source.sft_train \
  --data_dir my_gpt2/source/datasets/gsm8k_sft_examples/gsm8k_cot \
  --prefix gsm8k_cot \
  --pretrain_ckpt my_gpt2/results/sft_ultrachat/best.pt \
  --out_dir my_gpt2/results/sft_gsm8k_cot \
  --max_steps 1170 \
  --eval_every 200 \
  --max_train_examples -1 \
  --save_every 500


echo "Job completed."

# Slurm Commands
# module load devel/python/3.10.5
# sbatch -> Run script
# squeue -> show the list
# scancel <job_id> -> cancel job
# sinfo_t_idle
# squeue --start

# conda activate llm-env
# salloc --partition=dev_gpu_h100 --ntasks=1 --time=30 --mem=5000 --gres=gpu:1
# python -m my_gpt2.source.train
# torchrun --standalone --nproc_per_node=1 -m my_gpt2.source.train

# Install Kernel: python -m ipykernel install --user --name llm-env --display-name "Python (llm-env)"