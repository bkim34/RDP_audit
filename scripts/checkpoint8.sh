
for EPS in 10.0 17.85
do
    for SEED in 15 16 17 18 19 20 21 22 23 24
    do
        for MAX_GRAD_NORM in 1.0
        do
            # MNIST
            BASE_OUT="exp_data/max_grad_norm/${MAX_GRAD_NORM}/seed${SEED}"
            FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
            if [ -f "${FINAL_OUT}/done.txt" ]; then
                echo "Skipping ${FINAL_OUT}"
            else
                mkdir -p "${BASE_OUT}"
                python3 audit_model.py --data_name mnist_half --model_name cnn --n_epochs 100 --lr 1.33e-4 \
                    --max_grad_norm "$MAX_GRAD_NORM" --epsilon "$EPS" --seed "$SEED" \
                    --fixed_init pretrained_models/cnn_mnist_half.pt \
                    --out "${BASE_OUT}" --block_size 30000
                touch "${FINAL_OUT}/done.txt"
            fi

            # CIFAR-10
            BASE_OUT="exp_data/max_grad_norm/${MAX_GRAD_NORM}/seed${SEED}"
            FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
            if [ -f "${FINAL_OUT}/done.txt" ]; then
                echo "Skipping ${FINAL_OUT}"
            else
                mkdir -p "${BASE_OUT}"
                python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df 1000 --n_epochs 200 --lr 4e-5 \
                    --max_grad_norm "$MAX_GRAD_NORM" --epsilon "$EPS" --seed "$SEED" \
                    --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
                    --out "${BASE_OUT}" --block_size 10000
                touch "${FINAL_OUT}/done.txt"
            fi
        done
    done
done
