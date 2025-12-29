# Chinese-English Machine Translation Project

This project implements Chinese-English machine translation using both RNN-based and Transformer-based neural machine translation (NMT) models.

## Project Structure

```
hw3_code/
├── data/
│   ├── preprocessor.py          # Data preprocessing and tokenization
│   ├── vocabulary.py            # Vocabulary management
│   └── dataloader.py            # PyTorch DataLoader
├── models/
│   ├── rnn/
│   │   ├── attention.py         # Attention mechanisms (dot, multiplicative, additive)
│   │   ├── encoder.py           # RNN encoder
│   │   ├── decoder.py           # RNN decoder
│   │   └── seq2seq.py           # Complete Seq2Seq model
│   └── transformer/
│       ├── attention.py         # Multi-head attention
│       ├── positional_encoding.py  # Positional embeddings
│       ├── normalization.py     # LayerNorm and RMSNorm
│       ├── encoder.py           # Transformer encoder
│       ├── decoder.py           # Transformer decoder
│       └── transformer.py       # Complete Transformer model
├── utils/
│   ├── metrics.py               # BLEU-4 and precision_n metrics
│   ├── beam_search.py           # Beam search decoding
│   └── training_utils.py        # Training utilities
├── config.py                    # Configuration file
├── train_rnn.py                 # RNN training script
├── train_transformer.py         # Transformer training script
├── evaluate.py                  # Model evaluation and comparison
├── inference.py                 # One-click inference script
└── README.md                    # This file
```

## Requirements

Select mirror image "lepton-cci/ubuntu20.04-py3.8-pytorch2.3.1-cuda12.6.1-cudnn-devel" when creating CCI or ACP task.

<!-- ```bash
pip install torch torchvision
pip install jieba
pip install tqdm
``` -->

```bash
pip install -r requirements.txt
```

## Dataset
The initial dataset provided by the project is located in the `./dataset/initial/` directory with the following structure:
- `train_100k.jsonl` - Large training set (100k samples)
- `train_10k.jsonl` - Small training set (10k samples)
- `valid.jsonl` - Validation set (500 samples)
- `test.jsonl` - Test set (200 samples)

Each line is a JSON object with format: `{"en": "...", "zh": "...", "index": ...}`

If you want to use the dataset uploaded to the 2025 SLAI NLP&LLMs group on December 26th (reportedly of higher quality), please refer to the data in the directory under `./dataset/`:
- `train_100k_retranslated_hunyuan.jsonl` - Large training set (100k samples)
- `train_mixed_v2.jsonl` - Small training set (50k samples)
- `valid_retranslated_hunyuan.jsonl` - Validation set (500 samples)
- `test_retranslated_hunyuan.jsonl` - Test set (200 samples)

Each line is a JSON object with format: `{"en": "...", "zh": "...", "index": ..., "zh_hy": ...}`

When using different datasets, pay attention to the input parameter `zh_key` in the `preprocessor.load_and_preprocess` function within `train_xxx.py`.

## Usage

### 1. Train RNN Model

Since switching to multi-GPU training, I can no longer run command like `python train_rnn.py` anymore.

**Basic training (small dataset):**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py
```

**Train with different attention mechanisms:**
```bash
# Dot-product attention
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention dot

# Multiplicative attention
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention multiplicative

# Additive (Bahdanau) attention
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention additive
```

**Train with different RNN cells:**
GRU is currently unusable and there's no need to use it.

**Adjust teacher forcing ratio:**
```bash
# Full teacher forcing
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 1.0

# No teacher forcing (free running)
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 0.0

# Scheduled sampling
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 0.5
```

**Train on large dataset:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --use-large --epochs 20
```

**Custom hyperparameters:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --batch-size 128 --lr 0.0005 --epochs 30
```

### 2. Train Transformer Model

**Basic training:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py
```

**Architectural ablations:**
```bash
# Different positional embeddings
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --position-embedding sinusoidal
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --position-embedding learned
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --position-embedding relative

# Different normalization
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --norm-type LayerNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --norm-type RMSNorm

# Combined
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --position-embedding learned --norm-type RMSNorm
```

**Hyperparameter sensitivity:**
```bash
# Different batch sizes
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --batch-size 32
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --batch-size 64
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --batch-size 128

# Different learning rates
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --lr 0.0001
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --lr 0.0003
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --lr 0.0005

# Different model scales
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --d-model 256 --num-layers 2  # Small
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --d-model 512 --num-layers 4  # Medium
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --d-model 512 --num-layers 6  # Large
```

**Train on large dataset with label smoothing:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --use-large --label-smoothing --epochs 20
```

### 3. Finetune Pretrained Model
The finetuning code doesn't support multi-GPU training, just run `python XXX` directly:
```bash
export HD_ENDPOINT="https://hf-mirror.com"  # will download model and tokenizer from Huggingface
python finetune.py  --use-large  # using google/mt5-small default
```

### 4. One-Click Inference
The `inference.py` script provides easy-to-use translation functionality:

**Interactive mode:**
```bash
python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt
```

**Translate single sentence:**
```bash
python inference.py --model transformer --checkpoint ./checkpoints/transformer/best_model.pt --text "你好世界"
```

**Translate file with beam search:**
```bash
python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt --input input.txt --output output.txt --beam-size 5
```

### 5. Evaluate Models

**Evaluate single model:**
```bash
# Evaluate RNN
python -m torch.distributed.launch --nproc_per_node=1 --use_env evaluate.py --rnn-checkpoint ./checkpoints/rnn/best_model_dot_LSTM.pt

# Evaluate Transformer
python -m torch.distributed.launch --nproc_per_node=1 --use_env evaluate.py --transformer-checkpoint ./checkpoints/transformer/sinusoidal_LayerNorm_d512_l4_bs64_lr0.001/best_model.pt
```

**Compare RNN vs Transformer:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env evaluate.py \
    --rnn-checkpoint ./checkpoints/rnn/best_model_dot_LSTM.pt \
    --transformer-checkpoint ./checkpoints/transformer/sinusoidal_LayerNorm_d512_l4_bs64_lr0.001/best_model.pt \
    --output comparison_results.json
```

## Experiments

### RNN Experiments

1. **Attention Mechanism Comparison:**
   - Train models with dot, multiplicative, and additive attention
   - Compare BLEU scores and attention visualization

2. **Training Policy Comparison:**
   - Teacher forcing ratio: 1.0, 0.5, 0.0
   - Analyze impact on convergence and final performance

3. **Decoding Strategy:**
   - Greedy vs Beam search (beam sizes: 3, 5, 10)
   - Analyze trade-off between speed and quality

### Transformer Experiments

1. **Architectural Ablations:**
   - Position embeddings: sinusoidal, learned, relative
   - Normalization: LayerNorm, RMSNorm
   - Compare 9 combinations (3x3)

2. **Hyperparameter Sensitivity:**
   - Batch size: 32, 64, 128
   - Learning rate: 1e-4, 3e-4, 5e-4
   - Model scale: Small (2 layers), Medium (4 layers), Large (6 layers)

3. **Dataset Size Impact:**
   - Train on 10k vs 100k samples
   - Analyze data efficiency

### Model Comparison

Compare RNN and Transformer on:
1. **Architecture:**
   - Sequential vs parallel computation
   - Recurrence vs self-attention
   - Parameter count

2. **Training Efficiency:**
   - Training time per epoch
   - Convergence speed
   - GPU memory usage

3. **Translation Performance:**
   - BLEU-4 score
   - Precision-n (n=1,2,3,4)
   - Qualitative analysis

4. **Inference:**
   - Latency per sentence
   - Scalability to long sequences

5. **Practical Trade-offs:**
   - Model size vs performance
   - Training cost vs inference cost
   - Ease of implementation

## Configuration

Edit `config.py` to change default hyperparameters:
- Dataset paths
- Model architecture parameters
- Training hyperparameters
- Evaluation settings

## Output

Training outputs are saved to:
- `./checkpoints/rnn/` - RNN model checkpoints
- `./checkpoints/transformer/` - Transformer model checkpoints

Each checkpoint directory contains:
- `best_model.pt` - Best model based on validation BLEU
- `checkpoint_epoch_*.pt` - Periodic checkpoints
- `vocab_zh.pkl` - Chinese vocabulary
- `vocab_en.pkl` - English vocabulary
- `results.txt` - Evaluation results and sample translations

## Metrics

The following metrics are computed:
- **BLEU-4:** Standard machine translation metric based on NLTK (range 0-1)
- **Precision-n:** N-gram precision for n=1,2,3,4 based on NLTK
- **Inference time:** Average time per sentence

## Tips

1. **Quick Testing:** Use small dataset (10k) for rapid experimentation
2. **GPU Memory:** Reduce batch size if running out of memory
3. **Convergence:** Monitor validation BLEU every epoch
4. **Overfitting:** Use dropout and early stopping
5. **Reproducibility:** Set random seed in config

## Troubleshooting

**Out of memory:**
- Reduce batch size
- Reduce model size (d_model, num_layers)
- Use gradient accumulation

**Slow training:**
- Increase batch size
- Use smaller dataset for debugging
- Ensure CUDA is available

**Poor performance:**
- Train for more epochs
- Increase model capacity
- Use larger training set
- Try different hyperparameters

## Citation

This project implements techniques from:
- Bahdanau et al. (2014) - Neural Machine Translation by Jointly Learning to Align and Translate
- Vaswani et al. (2017) - Attention Is All You Need
- Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding

## License

MIT License
