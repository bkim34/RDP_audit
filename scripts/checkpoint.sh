#!/bin/bash
set -euo pipefail

for EPS in 2.0 4.38 6.57 10.0 17.85
do
    for SEED in 5 6 7 8 9 10 11 12 13 14 #15 16 17 18 19 20 21 22 23 24
    do
        # # MNIST
        # ## average-case
        # BASE_OUT="exp_data/model_init/fixed_average/seed${SEED}"
        # FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
        # if [ -f "${FINAL_OUT}/done.txt" ]; then
        #     echo "Skipping ${FINAL_OUT}"
        # else
        #     mkdir -p "${BASE_OUT}"
        #     python3 audit_model.py --data_name mnist_half --model_name cnn --lr 1.33e-4 --epsilon "$EPS" \
        #         --fixed_init \
        #         --seed "$SEED" --out "${BASE_OUT}" --block_size 30000
        #     touch "${FINAL_OUT}/done.txt"
        # fi

        # ## worst-case
        # BASE_OUT="exp_data/model_init/fixed_worst/seed${SEED}"
        # FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
        # if [ -f "${FINAL_OUT}/done.txt" ]; then
        #     echo "Skipping ${FINAL_OUT}"
        # else
        #     mkdir -p "${BASE_OUT}"
        #     python3 audit_model.py --data_name mnist_half --model_name cnn --n_reps 200 --lr 1.33e-4 --epsilon "$EPS" \
        #         --fixed_init pretrained_models/cnn_mnist_half.pt \
        #         --seed "$SEED" --out "${BASE_OUT}" --block_size 30000
        #     touch "${FINAL_OUT}/done.txt"
        # fi

        # CIFAR-10
        ## average-case
        BASE_OUT="exp_data/model_init/fixed_average/seed${SEED}"
        FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
        if [ -f "${FINAL_OUT}/done.txt" ]; then
            echo "Skipping ${FINAL_OUT}"
        else
            mkdir -p "${BASE_OUT}"
            python3 audit_model.py --data_name cifar10_half --model_name cnn --n_epochs 200 --lr 8e-5 --epsilon "$EPS" \
                --fixed_init \
                --seed "$SEED" --out "${BASE_OUT}" --block_size 10000
            touch "${FINAL_OUT}/done.txt"
        fi

        # worst-case
        BASE_OUT="exp_data/model_init/fixed_worst/seed${SEED}"
        FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
        if [ -f "${FINAL_OUT}/done.txt" ]; then
            echo "Skipping ${FINAL_OUT}"
        else
            mkdir -p "${BASE_OUT}"
            python3 audit_model.py --data_name cifar10_half --model_name cnn --n_epochs 200 --lr 4e-5 --epsilon "$EPS" \
                --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
                --seed "$SEED" --out "${BASE_OUT}" --block_size 10000
            touch "${FINAL_OUT}/done.txt"
        fi
    done
done


# for EPS in 2.0 4.38 6.57 10.0 17.85
# do
#     for SEED in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
#     do
#         for N_SAMPLES in 100 1000
#         do
#             # MNIST
#             BASE_OUT="exp_data/dataset_size/${N_SAMPLES}samples/seed${SEED}"
#             FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
#             if [ -f "${FINAL_OUT}/done.txt" ]; then
#                 echo "Skipping ${FINAL_OUT}"
#             else
#                 mkdir -p "${BASE_OUT}"
#                 python3 audit_model.py --data_name mnist_half --model_name cnn --n_df "$N_SAMPLES" --lr 1.33e-4 \
#                     --n_epochs 100 --epsilon "$EPS" --seed "$SEED" \
#                     --fixed_init pretrained_models/cnn_mnist_half.pt \
#                     --out "${BASE_OUT}"
#                 touch "${FINAL_OUT}/done.txt"
#             fi

#             # CIFAR-10
#             BASE_OUT="exp_data/dataset_size/${N_SAMPLES}samples/seed${SEED}"
#             FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
#             if [ -f "${FINAL_OUT}/done.txt" ]; then
#                 echo "Skipping ${FINAL_OUT}"
#             else
#                 mkdir -p "${BASE_OUT}"
#                 python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df "$N_SAMPLES" --lr 4e-5 \
#                     --n_epochs 200 --epsilon "$EPS" --seed "$SEED" \
#                     --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
#                     --out "${BASE_OUT}"
#                 touch "${FINAL_OUT}/done.txt"
#             fi
#         done
#     done
# done

# # ensure model_init.sh is run first
# # copy results from worst-case as full dataset
# # if [ ! -d exp_data/dataset_size/-1samples ]; then
# #     cp -r exp_data/model_init/fixed_worst exp_data/dataset_size/-1samples
# # fi

# for EPS in 2.0 4.38 6.57 10.0 17.85
# do
#     for SEED in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
#     do
#         for MAX_GRAD_NORM in 0.1 10.0
#         do
#             # MNIST
#             BASE_OUT="exp_data/max_grad_norm/${MAX_GRAD_NORM}/seed${SEED}"
#             FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
#             if [ -f "${FINAL_OUT}/done.txt" ]; then
#                 echo "Skipping ${FINAL_OUT}"
#             else
#                 mkdir -p "${BASE_OUT}"
#                 python3 audit_model.py --data_name mnist_half --model_name cnn --n_epochs 100 --lr 1.33e-4 \
#                     --max_grad_norm "$MAX_GRAD_NORM" --epsilon "$EPS" --seed "$SEED" \
#                     --fixed_init pretrained_models/cnn_mnist_half.pt \
#                     --out "${BASE_OUT}" --block_size 30000
#                 touch "${FINAL_OUT}/done.txt"
#             fi

#             # CIFAR-10
#             BASE_OUT="exp_data/max_grad_norm/${MAX_GRAD_NORM}/seed${SEED}"
#             FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
#             if [ -f "${FINAL_OUT}/done.txt" ]; then
#                 echo "Skipping ${FINAL_OUT}"
#             else
#                 mkdir -p "${BASE_OUT}"
#                 python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df 1000 --n_epochs 200 --lr 4e-5 \
#                     --max_grad_norm "$MAX_GRAD_NORM" --epsilon "$EPS" --seed "$SEED" \
#                     --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
#                     --out "${BASE_OUT}" --block_size 10000
#                 touch "${FINAL_OUT}/done.txt"
#             fi
#         done
#     done
# done

# # if [ ! -d exp_data/max_grad_norm/1.0 ]; then
# #     cp -r exp_data/model_init/fixed_worst exp_data/max_grad_norm/1.0
# # fi