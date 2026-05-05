#!/bin/bash
set -euo pipefail

cd ~/bb-audit-dpsgd-renyi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bb_audit_dpsgd

python scripts/make_renyi_graphs.py \
    --exp-data-root exp_data \
    --alphas 3 5 \
    --seeds 5 6 7 8 9 \
    --delta 1e-5 \
    --lr 1e-4 \
    --epochs-cifar 250 \
    --epochs-mnist 750 \
    --batch-size 400 \
    --ema-rate 1.0 \
    --device auto \
    --require-cuda
