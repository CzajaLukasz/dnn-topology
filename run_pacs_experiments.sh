#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
EPOCHS_LIST="1 10 25 50 75 100"
SPLIT=1024
THRESHOLD="0.5"
TOTAL_EPOCHS=100

MODELS=("convnext_tiny")
DOMAINS_TEST=("pacs_art" "pacs_cartoon" "pacs_sketch")

TRIALS_SOURCE=(
    "1:0.0"
    "2:0.1"
    "3:0.2"
    "4:0.3"
    "5:0.4"
    "6:0.5"
)

for NET in "${MODELS[@]}"; do
    echo "=========================================================="
    echo "START: $NET na PACS"
    echo "=========================================================="
    
    mkdir -p ./checkpoint/${NET}_pacs_photo/
    mkdir -p ./losses/${NET}_pacs_photo/
    
    # 1. Trening na domenie źródłowej (photo) - 6 poziomów szumu
    for ITEM in "${TRIALS_SOURCE[@]}"; do
        TRIAL="${ITEM%%:*}"
        PERM="${ITEM##*:}"
        
        echo "----------------------------------------------------------"
        echo "[$NET | PACS Photo] Trial $TRIAL (Szum: $PERM)"
        echo "----------------------------------------------------------"
        
        python main.py \
            --train 1 \
            --build_graph 1 \
            --net "$NET" \
            --dataset pacs_photo \
            --trial "$TRIAL" \
            --permute_labels "$PERM" \
            --n_epochs_train "$TOTAL_EPOCHS" \
            --epochs_test "$EPOCHS_LIST" \
            --graph_type functional_big_networks \
            --split "$SPLIT" \
            --thresholds "$THRESHOLD"
    done
    
    # 2. Ewaluacja domen docelowych OOD (art, cartoon, sketch)
    for DOMAIN in "${DOMAINS_TEST[@]}"; do
        echo "----------------------------------------------------------"
        echo "[$NET | Transfer OOD -> $DOMAIN] Trial 1 (Szum: 0.0)"
        echo "----------------------------------------------------------"
        
        mkdir -p ./checkpoint/${NET}_${DOMAIN}/
        mkdir -p ./losses/${NET}_${DOMAIN}/
        
        python main.py \
            --train 1 \
            --build_graph 1 \
            --net "$NET" \
            --dataset "$DOMAIN" \
            --trial 1 \
            --permute_labels 0.0 \
            --n_epochs_train "$TOTAL_EPOCHS" \
            --epochs_test "$EPOCHS_LIST" \
            --graph_type functional_big_networks \
            --split "$SPLIT" \
            --thresholds "$THRESHOLD"
    done
done

echo "Wszystkie eksperymenty PACS dla ResNet-18 i ConvNeXt-Tiny zakończone!"