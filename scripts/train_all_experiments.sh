#!/bin/bash
# ============================================================================
# Comprehensive Training Script for NLP Course Project
# OPTIMIZED FOR 8x H100 GPU CLUSTER
# ============================================================================
#
# Hardware: 8x NVIDIA H100
# Estimated total time: ~3-5 hours (parallelized)
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Optimized for 8x H100
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# Larger batch sizes for multi-GPU (effective batch = BATCH_SIZE * NUM_GPUS)
BATCH_SIZE=64           # Per GPU, so effective batch = 64 * 8 = 512
EPOCHS_SMALL=10
EPOCHS_MAIN=15
LR=0.001

# Create directories
mkdir -p ./checkpoints/rnn
mkdir -p ./checkpoints/transformer
mkdir -p ./checkpoints/finetune
mkdir -p ./logs

echo "============================================================================"
echo "NLP Course Project - Multi-GPU Training (8x H100)"
echo "Started at: $(date)"
echo "============================================================================"

# ============================================================================
# PART 1: RNN-based NMT Experiments (15%)
# Note: RNN training is sequential, but we can run 3 attention types in parallel
# ============================================================================

echo ""
echo "============================================================================"
echo "PART 1: RNN Experiments (Running 3 attention types in PARALLEL)"
echo "============================================================================"

# Run all 3 attention types in parallel on different GPUs
echo "[1.1] Starting RNN training with all attention types in parallel..."

# DOT attention on GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29500 --use_env train_rnn.py \
    --attention dot \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS_MAIN \
    --teacher-forcing 1.0 \
    2>&1 | tee ./logs/rnn_dot.log &
PID_DOT=$!

# MULTIPLICATIVE attention on GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29501 --use_env train_rnn.py \
    --attention multiplicative \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS_MAIN \
    --teacher-forcing 1.0 \
    2>&1 | tee ./logs/rnn_multiplicative.log &
PID_MULT=$!

# ADDITIVE attention on GPU 4,5
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29502 --use_env train_rnn.py \
    --attention additive \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS_MAIN \
    --teacher-forcing 1.0 \
    2>&1 | tee ./logs/rnn_additive.log &
PID_ADD=$!

# Wait for all attention experiments to complete
echo "Waiting for attention experiments to complete..."
wait $PID_DOT $PID_MULT $PID_ADD
echo "All attention experiments completed!"

# -----------------------------------------------------------------------------
# 1.2 Training Policy Comparison (Teacher Forcing ratios)
# Run in parallel on different GPU pairs
# -----------------------------------------------------------------------------

echo ""
echo "[1.2] Training Policy Comparison (Teacher Forcing ratios in PARALLEL)..."

# TF = 0.5 on GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29500 --use_env train_rnn.py \
    --attention additive \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS_MAIN \
    --teacher-forcing 0.5 \
    2>&1 | tee ./logs/rnn_tf_0.5.log &
PID_TF05=$!

# TF = 0.0 on GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29501 --use_env train_rnn.py \
    --attention additive \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS_MAIN \
    --teacher-forcing 0.0 \
    2>&1 | tee ./logs/rnn_tf_0.0.log &
PID_TF00=$!

wait $PID_TF05 $PID_TF00
echo "RNN experiments completed!"

# ============================================================================
# PART 2: Transformer-based NMT Experiments (25%)
# ============================================================================

echo ""
echo "============================================================================"
echo "PART 2: Transformer Experiments"
echo "============================================================================"

# -----------------------------------------------------------------------------
# 2.1 Architectural Ablation (6 combinations)
# Run 4 at a time (2 GPUs each)
# -----------------------------------------------------------------------------

echo ""
echo "[2.1] Architectural Ablation (Batch 1: 4 experiments in parallel)..."

# sinusoidal + LayerNorm
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29500 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_sinusoidal_LayerNorm.log &
P1=$!

# sinusoidal + RMSNorm
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29501 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type RMSNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_sinusoidal_RMSNorm.log &
P2=$!

# learned + LayerNorm
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29502 --use_env train_transformer.py \
    --position-embedding learned --norm-type LayerNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_learned_LayerNorm.log &
P3=$!

# learned + RMSNorm
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=29503 --use_env train_transformer.py \
    --position-embedding learned --norm-type RMSNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_learned_RMSNorm.log &
P4=$!

wait $P1 $P2 $P3 $P4

echo "[2.1] Architectural Ablation (Batch 2: 2 remaining experiments)..."

# relative + LayerNorm
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29500 --use_env train_transformer.py \
    --position-embedding relative --norm-type LayerNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_relative_LayerNorm.log &
P5=$!

# relative + RMSNorm
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29501 --use_env train_transformer.py \
    --position-embedding relative --norm-type RMSNorm \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_MAIN --label-smoothing \
    2>&1 | tee ./logs/transformer_relative_RMSNorm.log &
P6=$!

wait $P5 $P6

# -----------------------------------------------------------------------------
# 2.2 Hyperparameter Sensitivity (Use all 8 GPUs for faster training)
# -----------------------------------------------------------------------------

echo ""
echo "[2.2] Hyperparameter Sensitivity Experiments..."

# 2.2.1 Batch Size (run 2 in parallel)
echo "[2.2.1] Batch Size experiments..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29500 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --batch-size 32 --lr $LR --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_bs_32.log &
P1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29501 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --batch-size 128 --lr $LR --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_bs_128.log &
P2=$!

wait $P1 $P2

# 2.2.2 Learning Rate (run 2 in parallel)
echo "[2.2.2] Learning Rate experiments..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29500 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --batch-size $BATCH_SIZE --lr 0.0001 --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_lr_0.0001.log &
P1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29501 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --batch-size $BATCH_SIZE --lr 0.0005 --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_lr_0.0005.log &
P2=$!

wait $P1 $P2

# 2.2.3 Model Scale (run 2 in parallel)
echo "[2.2.3] Model Scale experiments..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29500 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --d-model 256 --num-layers 2 \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_small.log &
P1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29501 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm \
    --d-model 512 --num-layers 6 \
    --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS_SMALL --label-smoothing \
    2>&1 | tee ./logs/transformer_large.log &
P2=$!

wait $P1 $P2

# -----------------------------------------------------------------------------
# 2.3 Pretrained Model Finetuning
# -----------------------------------------------------------------------------

echo ""
echo "[2.3] Pretrained Model Finetuning (mT5-small)..."
export HF_ENDPOINT="https://hf-mirror.com"
# Finetuning doesn't support multi-GPU in current implementation
CUDA_VISIBLE_DEVICES=0 python finetune.py --epochs 5 2>&1 | tee ./logs/finetune_mt5.log

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "Finished at: $(date)"
echo "============================================================================"
echo ""
echo "Experiment Summary:"
echo "-------------------"
echo "RNN Experiments (5 models):"
echo "  - 3 Attention types: dot, multiplicative, additive"
echo "  - 3 Teacher Forcing: 1.0, 0.5, 0.0"
echo ""
echo "Transformer Experiments:"
echo "  - 6 Architecture combinations (3 pos_emb x 2 norm)"
echo "  - 2 Batch sizes: 32, 128"
echo "  - 2 Learning rates: 0.0001, 0.0005"
echo "  - 2 Model scales: small, large"
echo ""
echo "Finetuning:"
echo "  - mT5-small"
echo ""
echo "Checkpoints: ./checkpoints/"
echo "Logs: ./logs/"
echo "============================================================================"
