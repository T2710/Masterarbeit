#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs_dataloader_%j.out
#SBATCH --error=logs/Logs_dataloader_%j.err
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=190G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100

echo "Job started ..."

source ~/.bashrc
conda activate llm-env

cd /pfs/data6/home/ul/ul_student/ul_raf24/project/Masterarbeit

export PYTHONUNBUFFERED=1
mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# python -m my_gpt2.source.dataloader --dataset edu_fineweb100B --max_tokens 100000000000
# python -m my_gpt2.source.ultrachat --out_dir my_gpt2/source/datasets/ultrachat_sft_examples --num_examples -1 --shard_size_examples 10000

# gsm8k dataset 
# python my_gpt2/source/gsm8k.py \
#   --out_dir my_gpt2/source/datasets/gsm8k_ab \
#   --mode both \
#   --val_fraction 0.1 \
#   --shard_size_examples 1000 \
#   --max_seq_len 1024 \
#   --supervise_eos \
#   --overwrite

# python my_gpt2/source/dataloader.py \
#   --dataset openwebmath \
#   --base_dir my_gpt2/source/datasets \
#   --max_tokens 5000000000


# orca_math dataset
python -m my_gpt2.source.orca_math \
  --out_dir my_gpt2/source/datasets/orca_math \
  --num_examples 100000 \
  --make_val_from_train \
  --val_fraction 0.1 \
  --max_seq_len 1024 \
  --mode both \
  --supervise_eos \
  --overwrite


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
