@echo off
REM Quick training script for the NLP course project (Windows)
REM This script trains minimal models quickly for demonstration purposes

echo ==============================================
echo NLP Project Quick Training Script (Windows)
echo ==============================================

REM Create checkpoint directories
if not exist ".\checkpoints\rnn" mkdir ".\checkpoints\rnn"
if not exist ".\checkpoints\transformer" mkdir ".\checkpoints\transformer"

echo.
echo Step 1: Training RNN Model (Additive Attention)
echo ==============================================
REM Train RNN with additive attention
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention additive ^
    --batch-size 64 ^
    --lr 0.001 ^
    --epochs 10 ^
    --teacher-forcing 0.5

echo.
echo Step 2: Training Transformer Model (Sinusoidal + LayerNorm)
echo ==============================================
REM Train Transformer with default settings
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal ^
    --norm-type LayerNorm ^
    --batch-size 64 ^
    --lr 0.001 ^
    --epochs 10 ^
    --label-smoothing

echo.
echo ==============================================
echo Training completed!
echo Checkpoints saved in .\checkpoints\
echo ==============================================
pause
