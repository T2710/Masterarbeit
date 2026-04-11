#!/bin/bash
#SBATCH --job-name=RunSimulationResumeXL
#SBATCH --output=logs/Logs_Pretrain_XL100B_resume_%j.out
#SBATCH --error=logs/Logs_Pretrain_XL100B_resume_%j.err
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

set -euo pipefail

cd /pfs/data6/home/ul/ul_student/ul_raf24/project/Masterarbeit

export PYTHONUNBUFFERED=1
mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

forward_usr1() {
  if [[ -n "${TORCHRUN_PID:-}" ]] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
    echo "Forwarding USR1 to torchrun (pid $TORCHRUN_PID) ..."
    kill -USR1 "$TORCHRUN_PID" 2>/dev/null || true
    pkill -USR1 -P "$TORCHRUN_PID" 2>/dev/null || true
  fi
}

trap forward_usr1 USR1

torchrun --standalone --nproc_per_node=4 -m my_gpt2.source.pretrain \
  --model gpt2-xl \
  --resume_checkpoint my_gpt2/results/pretraining/log_pretraining_gpt2-xl/resume_latest.pt \
  --data_root my_gpt2/source/datasets/edu_fineweb100B \
  --log_dir my_gpt2/results/pretraining/log_pretraining \
  --tokens_target 100000000000 \
  --total_batch_size 524288 \
  --B 8 --T 1024 \
  --max_lr 2.5e-4 --warmup_steps 2000 \
  --eval_every 2000 --ckpt_every 5000 &
TORCHRUN_PID=$!
set +e
wait "$TORCHRUN_PID"
exit_code=$?
set -e
trap - USR1
if [[ $exit_code -ne 0 ]]; then
  echo "Job failed with exit code $exit_code."
  exit "$exit_code"
fi

echo "Job completed."
