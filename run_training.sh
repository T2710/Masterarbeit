#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs_Pretrain_Large15B_%j.out
#SBATCH --error=logs/Logs_Pretrain_Large15B_%j.err
#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=190G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_h100

echo "Job started ..."

source ~/.bashrc
conda activate llm-env

cd /pfs/data6/home/ul/ul_student/ul_raf24/project/Masterarbeit

export PYTHONUNBUFFERED=1
mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain --model gpt2 --max_steps 80000
# torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain --model gpt2-medium --max_steps 80000 --B 32
torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain \
  --model gpt2-large \
  --data_root my_gpt2/source/datasets/edu_fineweb100B \
  --tokens_target 15000000000 \
  --total_batch_size 524288 \
  --B 16 --T 1024 \
  --max_steps 80000 \
  --max_lr 3e-4 --warmup_steps 2000 \
  --eval_every 2000 --ckpt_every 5000



# Continued Pretraining
# python -m my_gpt2.source.pretrain \
#   --model gpt2-medium \
#   --init_checkpoint my_gpt2/results/pretraining/log_pretraining_gpt2-medium/gpt2-medium_model_75000.pt \
#   --data_root my_gpt2/source/datasets/openwebmath \
#   --tokens_target 5000000000 \
#   --max_lr 1e-4 \
#   --warmup_steps 1000 \
#   --total_batch_size 524288 \
#   --B 32 --T 1024 \
#   --log_dir my_gpt2/results/continued_pretraining_openwebmath \
#   --run_name gpt2-medium_owm_5b


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
# lfs quota -h -u $USER /home  - Speicherplatz anzeigen
# du -sh .    - Speicherplatz im aktuellen Verzeichnis anzeigen