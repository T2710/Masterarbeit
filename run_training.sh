#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs_Pretrain_Large100B_%j.out
#SBATCH --error=logs/Logs_Pretrain_Large100B_%j.err
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@300
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

forward_usr1() {
  if [[ -n "${TORCHRUN_PID:-}" ]] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
    echo "Forwarding USR1 to torchrun (pid $TORCHRUN_PID) ..."
    kill -USR1 "$TORCHRUN_PID" 2>/dev/null || true
    pkill -USR1 -P "$TORCHRUN_PID" 2>/dev/null || true
  fi
}

trap forward_usr1 USR1

# torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain --model gpt2 --max_steps 80000
# torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain --model gpt2-medium --max_steps 80000 --B 32
torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain \
  --model gpt2-large \
  --data_root my_gpt2/source/datasets/edu_fineweb100B \
  --log_dir my_gpt2/results/pretraining/log_pretraining \
  --tokens_target 100000000000 \
  --total_batch_size 524288 \
  --B 16 --T 1024 \
  --max_lr 2.5e-4 --warmup_steps 2000 \
  --eval_every 2000 --ckpt_every 5000 &
TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
exit_code=$?
trap - USR1
if [[ $exit_code -ne 0 ]]; then
  echo "Job failed with exit code $exit_code."
  exit "$exit_code"
fi
  # --max_steps 80000 \

# python -m my_gpt2.source.pretrain \
#   --model gpt2-large \
#   --init_checkpoint my_gpt2/results/pretraining/gpt2-large_model_25000.pt \
#   --data_root my_gpt2/source/datasets/edu_fineweb100B \
#   --tokens_target 15000000000 \
#   --max_lr 2.4610e-04 \
#   --warmup_steps 1000 \
#   --total_batch_size 524288 \
#   --B 16 --T 1024 \
#   --run_name gpt2-large

# Resume example:
# torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain \
#   --model gpt2-large \
#   --resume_checkpoint my_gpt2/results/pretraining/log_pretraining_gpt2-large/resume_latest.pt \
#   --data_root my_gpt2/source/datasets/edu_fineweb100B \
#   --tokens_target 100000000000 \
#   --total_batch_size 524288 \
#   --B 16 --T 1024 \
#   --max_lr 2.5e-4 --warmup_steps 2000 \
#   --eval_every 2000 --ckpt_every 5000




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
