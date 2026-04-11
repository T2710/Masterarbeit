#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/logs_reasoning_%j.out
#SBATCH --error=logs/logs_reasoning_%j.err
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



echo 'Direct_SFT on orca_mat (100k) and GSM8K: '
python -m my_gpt2.source.eval_gsm8k \
  --checkpoint my_gpt2/results/instruction_sft/gsm8k_direct/best.pt \
  --data_root my_gpt2/source/datasets/gsm8k_ab \
  --test_split test_direct \
  --model gpt2-medium \
  --max_new_tokens 64 \
  --num_save_examples 100 \
  --save_dir logs/gsm8k_direct_eval_100k


echo 'Direct_SFT on orca_math (200k) and GSM8K: '
python -m my_gpt2.source.eval_gsm8k \
  --checkpoint my_gpt2/results/instruction_sft/gsm8k_direct_200k/best.pt \
  --data_root my_gpt2/source/datasets/gsm8k_ab \
  --test_split test_direct \
  --model gpt2-medium \
  --max_new_tokens 64 \
  --num_save_examples 100 \
  --save_dir logs/gsm8k_direct_eval_200k


echo 'Direct_SFT on orca_math and CoT_SFT onGSM8K: '
python -m my_gpt2.source.eval_gsm8k \
  --checkpoint my_gpt2/results/instruction_sft/gsm8k_cot/best.pt \
  --data_root my_gpt2/source/datasets/gsm8k_ab \
  --test_split test_cot \
  --model gpt2-medium \
  --max_new_tokens 300 \
  --num_save_examples 100 \
  --save_dir logs/gsm8k_cot_eval

echo 'CoT_SFT on orca_math and GSM8K: '
python -m my_gpt2.source.eval_gsm8k \
  --checkpoint my_gpt2/results/instruction_sft/gsm8k_cotX2/best.pt \
  --data_root my_gpt2/source/datasets/gsm8k_ab \
  --test_split test_cot \
  --model gpt2-medium \
  --max_new_tokens 300 \
  --num_save_examples 100 \
  --save_dir logs/gsm8k_cotX2_eval


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