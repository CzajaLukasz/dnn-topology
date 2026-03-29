#!/bin/bash

# Zatrzymanie skryptu w przypadku błędu
set -e

echo "Rozpoczynam zautomatyzowane testy..."

# Test 1: Mała sieć, proste dane
echo "--- Uruchamiam LeNet na MNIST ---"
python main.py --net lenet --dataset mnist --n_epochs_train 5 --epochs_test "2 5" --trial 1

# Test 2: Klasyczna sieć, standardowe dane
echo "--- Uruchamiam AlexNet na CIFAR10 ---"
python main.py --net alexnet --dataset cifar10 --n_epochs_train 5 --epochs_test "5" --trial 1

# Test 3: Głęboka sieć, standardowe dane
echo "--- Uruchamiam ResNet na CIFAR10 ---"
python main.py --net resnet --dataset cifar10 --n_epochs_train 5 --epochs_test "1 4" --trial 1

echo "Wszystkie testy zakończone pomyślnie!"