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

set -euo pipefail

cd ~/bb-audit-dpsgd-renyi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bb_audit_dpsgd

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('device 0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in Slurm job'"

bash "$1"