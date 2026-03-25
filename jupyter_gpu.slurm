#!/bin/bash
#SBATCH --job-name=bb_jupyter
#SBATCH --account=bfkk-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-jupyter-%j.out
#SBATCH --error=slurm-jupyter-%j.err

set -euo pipefail

cd ~/bb-audit-dpsgd-renyi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bb_audit_dpsgd

PORT="${PORT:-8888}"

echo "HOSTNAME: $(hostname)"
echo "PORT: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
which python

python -c "import torch; print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('device 0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in Jupyter Slurm job'"

echo
echo "Start an SSH tunnel from your local machine with:"
echo "ssh -L ${PORT}:localhost:${PORT} <your_username>@<cluster_login_host> -J <your_username>@$(hostname)"
echo
echo "Then open the URL printed by Jupyter below in your local browser."
echo

jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}"
