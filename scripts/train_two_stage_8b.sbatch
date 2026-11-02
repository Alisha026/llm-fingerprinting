#!/usr/bin/env bash
#SBATCH --job-name=ntl-two-stage-8b
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/dss/dsshome1/02/ra48way2/llm-fingerprinting
#SBATCH --output=logs/two_stage_8b_%j.log
#SBATCH --error=logs/two_stage_8b_%j.log
#SBATCH --open-mode=append

set -euo pipefail
cd /dss/dsshome1/02/ra48way2/llm-fingerprinting
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
python_bin="$PWD/myenv/bin/python"
output_dir="artifacts/two_stage_8b_job_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

record_exit() {
    local status=$?
    trap - EXIT
    "$python_bin" -m src.record_experiments || true
    echo "Job ${SLURM_JOB_ID} finished with exit code ${status}; output: ${output_dir}"
    exit "$status"
}
trap record_exit EXIT

echo "Job: ${SLURM_JOB_ID}; host: $(hostname); output: ${output_dir}"
echo "Started: $(date --iso-8601=seconds)"
"$python_bin" -m src.two_stage_lora check --pool data/public_pool_v1/pool.parquet
srun --ntasks=1 "$python_bin" -u -m src.two_stage_lora train \
    --pool data/public_pool_v1/pool.parquet \
    --encoder artifacts/pretrained/qwen3_8b_base \
    --train data/processed/train.csv \
    --validation data/processed/validation.csv \
    --output "$output_dir"
