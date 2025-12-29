@echo off
REM ============================================================================
REM Comprehensive Training Script for NLP Course Project (Windows)
REM Covers ALL required experiments from the assignment
REM ============================================================================
REM
REM Requirements covered:
REM - RNN: 3 attention types, 3 teacher forcing ratios
REM - Transformer: Position embeddings x Normalizations, Hyperparameter sensitivity
REM - T5 Finetuning
REM
REM Estimated total time: ~20-30 hours on single GPU
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set EPOCHS_SMALL=10
set EPOCHS_MAIN=15
set BATCH_SIZE=64
set LR=0.001

REM Create directories
if not exist ".\checkpoints\rnn" mkdir ".\checkpoints\rnn"
if not exist ".\checkpoints\transformer" mkdir ".\checkpoints\transformer"
if not exist ".\checkpoints\finetune" mkdir ".\checkpoints\finetune"
if not exist ".\logs" mkdir ".\logs"

echo ============================================================================
echo NLP Course Project - Comprehensive Training Script
echo Started at: %date% %time%
echo ============================================================================

REM ============================================================================
REM PART 1: RNN-based NMT Experiments (15%)
REM ============================================================================

echo.
echo ============================================================================
echo PART 1: RNN-based NMT Experiments
echo ============================================================================

REM -----------------------------------------------------------------------------
REM 1.1 Attention Mechanism Comparison
REM -----------------------------------------------------------------------------

echo.
echo [1.1] Training RNN with DOT-PRODUCT Attention...
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention dot ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-forcing 1.0

echo.
echo [1.1] Training RNN with MULTIPLICATIVE Attention...
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention multiplicative ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-forcing 1.0

echo.
echo [1.1] Training RNN with ADDITIVE (Bahdanau) Attention...
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention additive ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-forcing 1.0

REM -----------------------------------------------------------------------------
REM 1.2 Training Policy Comparison
REM -----------------------------------------------------------------------------

echo.
echo [1.2] Training RNN with Teacher Forcing = 0.5 (Scheduled Sampling)...
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention additive ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-forcing 0.5

echo.
echo [1.2] Training RNN with Teacher Forcing = 0.0 (Free Running)...
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py ^
    --attention additive ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS_MAIN% ^
    --teacher-forcing 0.0

echo.
echo RNN experiments completed!

REM ============================================================================
REM PART 2: Transformer-based NMT Experiments (25%)
REM ============================================================================

echo.
echo ============================================================================
echo PART 2: Transformer-based NMT Experiments
echo ============================================================================

REM -----------------------------------------------------------------------------
REM 2.1 Architectural Ablation (6 combinations)
REM -----------------------------------------------------------------------------

echo.
echo [2.1] Architectural Ablation Experiments...

REM Sinusoidal + LayerNorm
echo Training Transformer: position=sinusoidal, norm=LayerNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM Sinusoidal + RMSNorm
echo Training Transformer: position=sinusoidal, norm=RMSNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type RMSNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM Learned + LayerNorm
echo Training Transformer: position=learned, norm=LayerNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding learned --norm-type LayerNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM Learned + RMSNorm
echo Training Transformer: position=learned, norm=RMSNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding learned --norm-type RMSNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM Relative + LayerNorm
echo Training Transformer: position=relative, norm=LayerNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding relative --norm-type LayerNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM Relative + RMSNorm
echo Training Transformer: position=relative, norm=RMSNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding relative --norm-type RMSNorm ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_MAIN% --label-smoothing

REM -----------------------------------------------------------------------------
REM 2.2 Hyperparameter Sensitivity
REM -----------------------------------------------------------------------------

echo.
echo [2.2] Hyperparameter Sensitivity Experiments...

REM Batch Size
echo [2.2.1] Batch Size = 32
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --batch-size 32 --lr %LR% --epochs %EPOCHS_SMALL% --label-smoothing

echo [2.2.1] Batch Size = 128
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --batch-size 128 --lr %LR% --epochs %EPOCHS_SMALL% --label-smoothing

REM Learning Rate
echo [2.2.2] Learning Rate = 0.0001
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --batch-size %BATCH_SIZE% --lr 0.0001 --epochs %EPOCHS_SMALL% --label-smoothing

echo [2.2.2] Learning Rate = 0.0005
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --batch-size %BATCH_SIZE% --lr 0.0005 --epochs %EPOCHS_SMALL% --label-smoothing

REM Model Scale
echo [2.2.3] Model Scale: SMALL (d=256, layers=2)
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --d-model 256 --num-layers 2 ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_SMALL% --label-smoothing

echo [2.2.3] Model Scale: LARGE (d=512, layers=6)
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py ^
    --position-embedding sinusoidal --norm-type LayerNorm ^
    --d-model 512 --num-layers 6 ^
    --batch-size %BATCH_SIZE% --lr %LR% --epochs %EPOCHS_SMALL% --label-smoothing

REM -----------------------------------------------------------------------------
REM 2.3 Pretrained Model Finetuning
REM -----------------------------------------------------------------------------

echo.
echo [2.3] Pretrained Model Finetuning (mT5-small)...
set HF_ENDPOINT=https://hf-mirror.com
python finetune.py --epochs 5

REM ============================================================================
REM Summary
REM ============================================================================

echo.
echo ============================================================================
echo ALL EXPERIMENTS COMPLETED!
echo Finished at: %date% %time%
echo ============================================================================
echo.
echo Experiment Summary:
echo -------------------
echo RNN Experiments:
echo   - 3 Attention types: dot, multiplicative, additive
echo   - 3 Teacher Forcing ratios: 1.0, 0.5, 0.0
echo.
echo Transformer Experiments:
echo   - 6 Architecture combinations (3 pos_emb x 2 norm)
echo   - 2 Batch sizes: 32, 128 (+ baseline 64)
echo   - 2 Learning rates: 0.0001, 0.0005 (+ baseline 0.001)
echo   - 2 Model scales: small, large (+ baseline medium)
echo.
echo Finetuning:
echo   - mT5-small finetuned
echo.
echo Checkpoints saved in: .\checkpoints\
echo ============================================================================

pause
