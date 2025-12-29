#!/bin/bash
# Quick training script for the NLP course project
# This script trains minimal models quickly for demonstration purposes
# Due to time constraints, focus on training representative models

echo "=============================================="
echo "NLP Project Quick Training Script"
echo "=============================================="

# Set environment
export CUDA_VISIBLE_DEVICES=0

# Create checkpoint directories
mkdir -p ./checkpoints/rnn
mkdir -p ./checkpoints/transformer

echo ""
echo "Step 1: Training RNN Model (Additive Attention)"
echo "=============================================="
# Train RNN with additive attention (the most common in papers)
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py \
    --attention additive \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 10 \
    --teacher-forcing 0.5

echo ""
echo "Step 2: Training Transformer Model (Sinusoidal + LayerNorm)"
echo "=============================================="
# Train Transformer with default settings
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py \
    --position-embedding sinusoidal \
    --norm-type LayerNorm \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 10 \
    --label-smoothing

echo ""
echo "=============================================="
echo "Training completed!"
echo "Checkpoints saved in ./checkpoints/"
echo "=============================================="
