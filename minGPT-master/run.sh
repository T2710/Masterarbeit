#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs%j.out
#SBATCH --error=logs/Logs%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=127G
#SBATCH --gres=gpu:1
#SBATCH --partition=dev_gpu_h100

echo "Job started ..."

source ~/.bashrc
conda activate llm-env

cd /pfs/data6/home/ul/ul_student/ul_raf24/github/Masterarbeit/minGPT-master

export PYTHONUNBUFFERED=1
mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -u projects/openThoughts-114k/train.py

echo "Job completed."

# Slurm Commands
# module load devel/python/3.10.5
# sbatch -> Run script
# squeue -> show the list
# scancel <job_id> -> cancel job
# sinfo_t_idle
# squeue --start

# conda activate llm-env
# $ salloc --partition=dev_gpu_h100 --ntasks=1 --time=30 --mem=5000 --gres=gpu:1