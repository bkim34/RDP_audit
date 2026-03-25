#!/bin/bash
#SBATCH --job-name=bb_audit
#SBATCH --account=bfkk-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd ~/bb-audit-dpsgd-renyi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bb_audit_dpsgd

bash scripts/checkpoint.sh