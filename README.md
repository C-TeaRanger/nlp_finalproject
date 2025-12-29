# Chinese-English Machine Translation Project

A comprehensive comparative study of RNN-based and Transformer-based neural architectures for Chinese-to-English machine translation, analyzing the impact of attention mechanisms, positional encodings, and training strategies on model performance.

**GitHub:** https://github.com/C-TeaRanger/nlp_finalproject.git

---

## 🏆 Key Results

| Model | Best Configuration | Validation BLEU |
|-------|-------------------|-----------------|
| **Transformer** | Small (d=256, layers=2) | **0.0568** |
| **Transformer** | Relative + RMSNorm | 0.0464 |
| **RNN** | Additive Attention, TF=1.0 | 0.0370 |

> **Finding:** Smaller Transformer models outperform larger ones on this dataset size, suggesting regularization benefits.

---

## 📁 Project Structure

```
nlp_project/
├── data/
│   ├── preprocessor.py          # Data preprocessing and tokenization
│   ├── vocabulary.py            # BPE vocabulary management
│   └── dataloader.py            # PyTorch DataLoader
├── models/
│   ├── rnn/
│   │   ├── attention.py         # Attention mechanisms (dot, multiplicative, additive)
│   │   ├── encoder.py           # Bidirectional RNN encoder
│   │   ├── decoder.py           # RNN decoder with attention
│   │   └── seq2seq.py           # Complete Seq2Seq model
│   └── transformer/
│       ├── attention.py         # Multi-head self-attention
│       ├── positional_encoding.py  # Sinusoidal/Learned/Relative embeddings
│       ├── normalization.py     # LayerNorm and RMSNorm
│       └── transformer.py       # Complete Transformer model
├── utils/
│   ├── metrics.py               # BLEU-4 and precision metrics
│   ├── beam_search.py           # Beam search decoding
│   └── training_utils.py        # Training utilities
├── scripts/
│   ├── train_all_experiments.sh # Run all experiments (8x H100 optimized)
│   ├── extract_results.py       # Extract results from logs
│   └── visualize_results.py     # Generate charts for report
├── config.py                    # Configuration file
├── train_rnn.py                 # RNN training script
├── train_transformer.py         # Transformer training script
├── evaluate.py                  # Model evaluation
├── inference.py                 # One-click inference script
├── finetune.py                  # mT5 fine-tuning script
├── report.md                    # Project report
└── README.md
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### One-Click Inference
```bash
# Interactive mode
python inference.py --model transformer --checkpoint ./checkpoints/transformer/best_model.pt

# Translate a sentence
python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt --text "你好世界"

# Batch translation with beam search
python inference.py --model transformer --checkpoint ./checkpoints/transformer/best_model.pt \
    --input input.txt --output output.txt --beam-size 5
```

---

## 🧪 Experiments

### RNN Experiments

**Attention Mechanism Comparison:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention dot
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention multiplicative
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --attention additive
```

**Training Policy (Teacher Forcing):**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 1.0  # Full TF
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 0.5  # Scheduled
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_rnn.py --teacher-forcing 0.0  # Free running
```

### Transformer Experiments

**Architecture Ablation (Position Encoding × Normalization):**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py \
    --position-embedding sinusoidal --norm-type LayerNorm
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py \
    --position-embedding relative --norm-type RMSNorm
```

**Hyperparameter Sensitivity:**
```bash
# Batch size
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --batch-size 32
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --batch-size 128

# Learning rate
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --lr 0.0001
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --lr 0.0005

# Model scale
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --d-model 256 --num-layers 2
python -m torch.distributed.launch --nproc_per_node=1 --use_env train_transformer.py --d-model 512 --num-layers 6
```

### Run All Experiments (Multi-GPU)
```bash
# For 8x H100 cluster
chmod +x scripts/train_all_experiments.sh
bash scripts/train_all_experiments.sh
```

---

## 📊 Results Extraction & Visualization

```bash
# Extract results from training logs
python scripts/extract_results.py --log-dir ./logs --output ./results/experiment_results.json

# Generate visualization charts
python scripts/visualize_results.py --results ./results/experiment_results.json --output-dir ./figures
```

**Generated Figures:**
- `rnn_attention_comparison.png` - Attention mechanism comparison
- `rnn_teacher_forcing_comparison.png` - Training policy comparison
- `transformer_architecture_heatmap.png` - Architecture ablation
- `transformer_hyperparameter_sensitivity.png` - Hyperparameter effects
- `model_comparison.png` - RNN vs Transformer

---

## 📈 Key Findings

1. **Attention Mechanisms:** Additive (Bahdanau) attention significantly outperforms dot-product and multiplicative for Chinese-English translation.

2. **Teacher Forcing:** Essential for RNN convergence; free running leads to poor performance.

3. **Position Encodings:** Relative position encoding achieves the best results for Transformers.

4. **Model Scale:** Smaller models generalize better on limited data (50k-100k samples).

5. **Overall:** Transformers outperform RNNs by ~54% (relative improvement in BLEU).

---

## 📂 Dataset

The project uses Chinese-English parallel corpus:

| Dataset | Samples | Location |
|---------|---------|----------|
| Training (Large) | 100k | `./dataset/train_100k_retranslated_hunyuan.jsonl` |
| Training (Small) | 50k | `./dataset/train_mixed_v2.jsonl` |
| Validation | 500 | `./dataset/valid_retranslated_hunyuan.jsonl` |
| Test | 200 | `./dataset/test_retranslated_hunyuan.jsonl` |

---

## 🔧 Configuration

Edit `config.py` to modify:
- Dataset paths
- Model hyperparameters (hidden dim, layers, heads)
- Training settings (batch size, learning rate, epochs)

---

## 📚 References

- Bahdanau et al. (2014) - *Neural Machine Translation by Jointly Learning to Align and Translate*
- Vaswani et al. (2017) - *Attention Is All You Need*
- Su et al. (2021) - *RoFormer: Enhanced Transformer with Rotary Position Embedding*

---

## 📄 License

MIT License
