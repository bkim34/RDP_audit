
for EPS in 2.0 4.38 6.57 10.0 17.85
do
    for SEED in 15 16 17 18 19 20 21 22 23 24
    do
        for N_SAMPLES in 100 1000
        do
            # MNIST
            BASE_OUT="exp_data/dataset_size/${N_SAMPLES}samples/seed${SEED}"
            FINAL_OUT="${BASE_OUT}/mnist_half_cnn_eps${EPS}"
            if [ -f "${FINAL_OUT}/done.txt" ]; then
                echo "Skipping ${FINAL_OUT}"
            else
                mkdir -p "${BASE_OUT}"
                python3 audit_model.py --data_name mnist_half --model_name cnn --n_df "$N_SAMPLES" --lr 1.33e-4 \
                    --n_epochs 100 --epsilon "$EPS" --seed "$SEED" \
                    --fixed_init pretrained_models/cnn_mnist_half.pt \
                    --out "${BASE_OUT}"
                touch "${FINAL_OUT}/done.txt"
            fi

            # CIFAR-10
            BASE_OUT="exp_data/dataset_size/${N_SAMPLES}samples/seed${SEED}"
            FINAL_OUT="${BASE_OUT}/cifar10_half_cnn_eps${EPS}"
            if [ -f "${FINAL_OUT}/done.txt" ]; then
                echo "Skipping ${FINAL_OUT}"
            else
                mkdir -p "${BASE_OUT}"
                python3 audit_model.py --data_name cifar10_half --model_name cnn --n_df "$N_SAMPLES" --lr 4e-5 \
                    --n_epochs 200 --epsilon "$EPS" --seed "$SEED" \
                    --fixed_init pretrained_models/cnn_cifar100_cifar10_half.pt \
                    --out "${BASE_OUT}"
                touch "${FINAL_OUT}/done.txt"
            fi
        done
    done
done