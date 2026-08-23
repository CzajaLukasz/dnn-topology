#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
EPOCHS_LIST="1 10 25 50 75 100"
SPLIT=1024
THRESHOLD="0.5"
TOTAL_EPOCHS=100
NET="convnext_tiny"

echo "=========================================================="
echo "START: ConvNeXt-Tiny (MNIST -> USPS)"
echo "=========================================================="

mkdir -p ./checkpoint/${NET}_mnist/
mkdir -p ./checkpoint/${NET}_usps/
mkdir -p ./losses/${NET}_mnist/
mkdir -p ./losses/${NET}_usps/

# 6 triali MNIST (od 0% do 50% permutacji etykiet)
TRIALS_MNIST=(
    "1:0.0"
    "2:0.1"
    "3:0.2"
    "4:0.3"
    "5:0.4"
    "6:0.5"
)

for ITEM in "${TRIALS_MNIST[@]}"; do
    TRIAL="${ITEM%%:*}"
    PERM="${ITEM##*:}"
    
    echo "----------------------------------------------------------"
    echo "[$NET | MNIST] Uruchamianie Trial $TRIAL (Szum: $PERM)"
    echo "----------------------------------------------------------"
    
    python main.py \
        --train 1 \
        --build_graph 1 \
        --net "$NET" \
        --dataset mnist \
        --trial "$TRIAL" \
        --permute_labels "$PERM" \
        --n_epochs_train "$TOTAL_EPOCHS" \
        --epochs_test "$EPOCHS_LIST" \
        --graph_type functional_big_networks \
        --split "$SPLIT" \
        --thresholds "$THRESHOLD"
done

# 1 trial USPS (Czysty zbiór, test transferu domeny)
echo "----------------------------------------------------------"
echo "[$NET | USPS] Uruchamianie Trial 1 (Szum: 0.0)"
echo "----------------------------------------------------------"

python main.py \
    --train 1 \
    --build_graph 1 \
    --net "$NET" \
    --dataset usps \
    --trial 1 \
    --permute_labels 0.0 \
    --n_epochs_train "$TOTAL_EPOCHS" \
    --epochs_test "$EPOCHS_LIST" \
    --graph_type functional_big_networks \
    --split "$SPLIT" \
    --thresholds "$THRESHOLD"

echo "=========================================================="
echo "Obliczenia dla ConvNeXt-Tiny zakończone sukcesem!"
echo "=========================================================="