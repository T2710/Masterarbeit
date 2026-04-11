#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs_sft_%j.out
#SBATCH --error=logs/Logs_sft_%j.err
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=127G
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

# Fine tuning the model on orac math dataset
# python -m my_gpt2.source.instruction_sft \
#   --model gpt2-medium \
#   --init_checkpoint my_gpt2/results/continued_pretraining_openwebmath/gpt2-medium_owm_5b/gpt2-medium_model_09536.pt \
#   --data_root my_gpt2/source/datasets/orca_math \
#   --train_split train_direct \
#   --val_split val_direct \
#   --output_root my_gpt2/results/instruction_sft \
#   --run_name orca_math_sft_200k \
#   --epochs 1 \
#   --batch_size 8 \
#   --grad_accum_steps 4 \
#   --max_lr 2e-5



# Directly fine-tune the model on GSM8K dataset without reasoning (direct answer)
python -m my_gpt2.source.instruction_sft \
  --model gpt2-medium \
  --init_checkpoint my_gpt2/results/instruction_sft/orca_math_sft_200k/best.pt \
  --data_root my_gpt2/source/datasets/gsm8k_ab \
  --train_split train_direct \
  --val_split val_direct \
  --output_root my_gpt2/results/instruction_sft \
  --run_name gsm8k_direct_200k \
  --epochs 4 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --max_lr 2e-5

#  Fine-tune the model on GSM8K dataset with reasoning (chain-of-thought)
# python -m my_gpt2.source.instruction_sft \
#   --model gpt2-medium \
#   --init_checkpoint my_gpt2/results/instruction_sft/orca_math_sft_cot/ckpt_02812.pt \
#   --data_root my_gpt2/source/datasets/gsm8k_ab \
#   --train_split train_cot \
#   --val_split val_cot \
#   --output_root my_gpt2/results/instruction_sft \
#   --run_name gsm8k_cotX2 \
#   --epochs 4 \
#   --batch_size 4 \
#   --grad_accum_steps 2 \
#   --max_lr 2e-5



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
