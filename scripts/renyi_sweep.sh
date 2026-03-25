#!/bin/bash
set -euo pipefail

cd ~/bb-audit-dpsgd-renyi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bb_audit_dpsgd

python scripts/renyi_sweep.py \
    --results-root exp_data/max_grad_norm/1.0 \
    --run-template 'cifar10_half_cnn_eps{eps}' \
    --eps-values 2.0 4.38 6.57 10.0 17.85 \
    --seeds 5 6 7 8 9 \
    --renyi-order 3.0 \
    --lr 1e-4 \
    --epochs 10 \
    --batch-size 5000 \
    --device auto \
    --require-cuda \
    --output-txt renyi_results/max_grad_norm_1.0_cifar10_half_cnn_seeds5_9.txt
