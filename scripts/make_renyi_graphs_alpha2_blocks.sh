#!/bin/bash
set -euo pipefail

cd ~/bb-audit-dpsgd-renyi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bb_audit_dpsgd

python scripts/make_renyi_graphs_alpha2_blocks.py \
    --exp-data-root exp_data \
    --delta 1e-5 \
    --epochs-cifar 250 \
    --epochs-mnist 750 \
    --batch-size 400 \
    --lr 1e-4 \
    --ema-rate 1.0 \
    --device auto \
    --require-cuda
