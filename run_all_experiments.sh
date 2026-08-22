#!/usr/bin/env bash
set -e

# Konfiguracja środowiska
export CUDA_VISIBLE_DEVICES=0
EPOCHS_LIST="1 10 25 50 75 100"
SPLIT=1024
THRESHOLD="0.5"
TOTAL_EPOCHS=100

echo "=========================================================="
echo "ROZPOCZĘCIE SERII POMIAROWEJ TDA / GENERALIZATION GAP"
echo "=========================================================="

# 1. Przygotowanie czystych katalogów wyjściowych
mkdir -p ./checkpoint/resnet18_mnist/
mkdir -p ./checkpoint/resnet18_usps/
mkdir -p ./losses/resnet18_mnist/
mkdir -p ./losses/resnet18_usps/

# ========================================================
# SEKCJA 1: MNIST (Zbiór uczący dla regresora - 6 poziomów szumu)
# ========================================================
# Definicja par: trial:poziom_szumu
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
    echo "[MNIST] Uruchamianie Trial $TRIAL (Szum etykiet: $PERM)"
    echo "----------------------------------------------------------"
    
    python main.py \
        --train 1 \
        --build_graph 1 \
        --net resnet18 \
        --dataset mnist \
        --trial "$TRIAL" \
        --permute_labels "$PERM" \
        --n_epochs_train "$TOTAL_EPOCHS" \
        --epochs_test "$EPOCHS_LIST" \
        --graph_type functional_big_networks \
        --split "$SPLIT" \
        --thresholds "$THRESHOLD"
done

# ========================================================
# SEKCJA 2: USPS (Zbiór ewaluacyjny OOD - 1 trial)
# ========================================================
echo "----------------------------------------------------------"
echo "[USPS] Uruchamianie Trial 1 (Szum etykiet: 0.0)"
echo "----------------------------------------------------------"

python main.py \
    --train 1 \
    --build_graph 1 \
    --net resnet18 \
    --dataset usps \
    --trial 1 \
    --permute_labels 0.0 \
    --n_epochs_train "$TOTAL_EPOCHS" \
    --epochs_test "$EPOCHS_LIST" \
    --graph_type functional_big_networks \
    --split "$SPLIT" \
    --thresholds "$THRESHOLD"

echo "=========================================================="
echo "WSZYSTKIE OBLICZENIA ZAKOŃCZONE SUKCESEM!"
echo "=========================================================="

# Automatyczne uruchomienie ewaluacji regresji na zebranych danych
echo "Generowanie raportu końcowego..."
python evaluate_regression.py