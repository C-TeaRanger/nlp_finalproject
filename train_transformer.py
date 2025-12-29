"""
Training script for Transformer-based machine translation
Supports architectural ablations and hyperparameter sensitivity analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import argparse
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.preprocessor import Preprocessor
from data.vocabulary import Vocabulary
from data.dataloader import get_dataloader
from models.transformer.transformer import Transformer
from utils.metrics import calculate_all_metrics
from utils.training_utils import (
    save_checkpoint, load_checkpoint, count_parameters,
    AverageMeter, Timer, print_model_info, WarmupScheduler
)
from utils.beam_search import greedy_decode, beam_search_decode

from data.vocabulary_bpe_en import BPEVocabularyEN
from data.vocabulary_bpe_zh import BPEVocabularyZH

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        """
        Args:
            logits: (batch_size * seq_len, vocab_size)
            target: (batch_size * seq_len)
        """
        # Create smoothed labels
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for pad and true label
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        # Mask padding
        mask = (target != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        # Compute KL divergence
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = self.criterion(log_probs, true_dist)

        # Normalize by number of non-padding tokens
        loss = loss / mask.sum()

        return loss


def train_epoch(model, dataloader, optimizer, criterion, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    timer = Timer()
    timer.start()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(pbar):
        # Move to device
        src = src.to(config.DEVICE)
        tgt = tgt.to(config.DEVICE)
        src_len = src_len.to(config.DEVICE)

        # Forward pass (exclude last token in target for input, first token for output)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        outputs = model(src, tgt_input)
        # outputs: (batch_size, tgt_len-1, vocab_size)

        # Compute loss
        output_dim = outputs.size(-1)
        outputs = outputs.reshape(-1, output_dim)  # (batch_size * (tgt_len-1), vocab_size)
        tgt_output = tgt_output.reshape(-1)  # (batch_size * (tgt_len-1))

        loss = criterion(outputs, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        losses.update(loss.item(), src.size(0))

        # Update progress bar
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{lr:.6f}'
        })

    if epoch % 5 == 0:
        # Print GPU memory usage
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            print(f"GPU {i}: {allocated:.1f}/{memory:.1f} GB")

    elapsed = timer.stop()
    return losses.avg, elapsed


def evaluate(model, dataloader, criterion, config, vocab_src, vocab_tgt, use_beam_search=False):
    """Evaluate on validation/test set"""
    model.eval()
    losses = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            src = src.to(config.DEVICE)
            tgt = tgt.to(config.DEVICE)
            src_len = src_len.to(config.DEVICE)

            # Forward pass for loss
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            outputs = model(src, tgt_input)

            # Compute loss
            output_dim = outputs.size(-1)
            outputs_flat = outputs.reshape(-1, output_dim)
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(outputs_flat, tgt_flat)
            losses.update(loss.item(), src.size(0))

            # Generate translations for BLEU
            for i in range(src.size(0)):
                src_seq = src[i:i+1]
                src_len_seq = src_len[i:i+1]

                if use_beam_search:
                    pred_tokens = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        config.BEAM_SIZE, config.MAX_DECODE_LENGTH, config.DEVICE
                    )
                else:
                    pred_tokens = greedy_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        config.MAX_DECODE_LENGTH, config.DEVICE
                    )

                # Get reference tokens (skip <bos> and <eos>)
                ref_tokens = tgt[i].cpu().tolist()
                ref_tokens = [t for t in ref_tokens if t not in [
                    vocab_tgt.pad_idx, vocab_tgt.bos_idx, vocab_tgt.eos_idx
                ]]

                # Convert to words
                ref_words = vocab_tgt.decode(ref_tokens, skip_special=True).split()
                pred_words = vocab_tgt.decode(pred_tokens, skip_special=True).split()

                references.append(ref_words)
                hypotheses.append(pred_words)

    # Calculate metrics
    metrics = calculate_all_metrics(references, hypotheses)
    metrics['loss'] = losses.avg

    return metrics, references, hypotheses


def dataset_prepare(data, vocab_zh, vocab_en, max_length=None, add_eos=False):
    "assume source is Chinese, target is English"
    zh_text, en_text = map(list, zip(*data))
    zh_tokens = vocab_zh.segement_text(zh_text, max_length=max_length, is_pretokenized=False, output_ids=True)
    en_tokens = vocab_en.segement_text(en_text, max_length=max_length, is_pretokenized=False, output_ids=True)
    # if add_eos:
    #     zh_tokens = [tokens + [vocab_zh.eos_token] for tokens in zh_tokens]
    #     en_tokens = [[vocab_en.bos_token] + tokens + [vocab_en.eos_token] for tokens in en_tokens]
    # else:
    #     en_tokens = [[vocab_en.bos_token] + tokens for tokens in en_tokens] 
    return list(zip(zh_tokens, en_tokens))


def main(args):
    import torch.distributed as dist

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # if local_rank == 0:
    #     print(f"Using device: {device}")

    print(f"Using device: {device}")

    config = Config()

    # Override config with command line arguments
    if args.position_embedding:
        config.POSITION_EMBEDDING = args.position_embedding
    if args.norm_type:
        config.NORM_TYPE = args.norm_type
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.d_model:
        config.TRANS_D_MODEL = args.d_model
    if args.num_layers:
        config.TRANS_NUM_ENCODER_LAYERS = args.num_layers
        config.TRANS_NUM_DECODER_LAYERS = args.num_layers
    if args.num_heads:
        config.TRANS_NHEAD = args.num_heads
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # config.DEVICE = device
    # if local_rank == 0:
    #     print(f"Using device: {device}")

    # # Set device
    # device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    # config.DEVICE = device
    # print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = config.TRANS_CHECKPOINT_DIR
    exp_name = f"{config.POSITION_EMBEDDING}_{config.NORM_TYPE}_d{config.TRANS_D_MODEL}_l{config.TRANS_NUM_ENCODER_LAYERS}_bs{config.BATCH_SIZE}_lr{config.LEARNING_RATE}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = Preprocessor()

    train_path = config.TRAIN_LARGE_PATH if args.use_large else config.TRAIN_SMALL_PATH
    train_data = preprocessor.load_and_preprocess(train_path, return_text=True)
    valid_data = preprocessor.load_and_preprocess(config.VALID_PATH, return_text=True)
    test_data = preprocessor.load_and_preprocess(config.TEST_PATH, return_text=True)

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build vocabularies
    print("\nBuilding vocabularies...")
    zh_sentences = [zh for zh, en in train_data]
    en_sentences = [en for zh, en in train_data]

    # breakpoint()

    # vocab_zh = Vocabulary("Chinese")
    # vocab_zh.build_vocab(zh_tokens, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)

    # vocab_en = Vocabulary("English")
    # vocab_en.build_vocab(en_tokens, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)

    # # using BPE
    vocab_zh = BPEVocabularyZH("Chinese")
    # vocab_zh.train_from_texts(zh_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
    # vocab_zh.save(os.path.join(checkpoint_dir, 'tokenizer_zh.pkl'), os.path.join(checkpoint_dir, 'vocab_zh.json'))

    vocab_en = BPEVocabularyEN("English")
    # vocab_en.train_from_texts(en_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
    # vocab_en.train_from_texts(en_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
    # vocab_en.save(os.path.join(checkpoint_dir, 'tokenizer_en.pkl'), os.path.join(checkpoint_dir, 'vocab_en.json'))

    if local_rank == 0:
        vocab_zh.train_from_texts(zh_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
        vocab_zh.save(os.path.join(checkpoint_dir, 'tokenizer_zh.pkl'), os.path.join(checkpoint_dir, 'vocab_zh.json'))
        vocab_en.train_from_texts(en_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
        vocab_en.save(os.path.join(checkpoint_dir, 'tokenizer_en.pkl'), os.path.join(checkpoint_dir, 'vocab_en.json'))
        print("✓ Tokenizers trained and saved")

    # Synchronize all processes
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Add delay for NFS caching
    # time.sleep(5)

    en_tokenzier_path = os.path.join(checkpoint_dir, 'tokenizer_en.pkl')
    zh_tokenzier_path = os.path.join(checkpoint_dir, 'tokenizer_zh.pkl')

    # All processes load the tokenizer with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not os.path.exists(en_tokenzier_path):
                raise FileNotFoundError(f"Tokenizer file not found: {en_tokenzier_path}")
            if not os.path.exists(zh_tokenzier_path):
                raise FileNotFoundError(f"Tokenizer file not found: {zh_tokenzier_path}")

            vocab_en.load(en_tokenzier_path)
            vocab_zh.load(zh_tokenzier_path)
            print(f"✓ Tokenizers loaded from rank {local_rank}")
            break

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠ Rank {local_rank}: Load attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"✗ Rank {local_rank}: Failed to load tokenizers after {max_retries} attempts")
                raise

    # # Save vocabularies
    # vocab_zh.save(os.path.join(checkpoint_dir, 'vocab_zh.pkl'))
    # vocab_en.save(os.path.join(checkpoint_dir, 'vocab_en.pkl'))

    # vocab_zh = BPEVocabularyZH("Chinese")
    # vocab_zh.load(os.path.join(checkpoint_dir, 'tokenizer_zh.pkl'))

    # vocab_en = BPEVocabularyEN("English")
    # vocab_en.load(os.path.join(checkpoint_dir, 'tokenizer_en.pkl'))
    print("Vocabulary loaded.")

    print("\nPreparing datasets...")
    start = time.time()
    test_data = dataset_prepare(test_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH)
    print(f"Tokenize testing dataset takes: {time.time() - start:.2f} seconds")

    start = time.time()
    valid_data = dataset_prepare(valid_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH)
    print(f"Tokenize validation dataset takes: {time.time() - start:.2f} seconds")

    start = time.time()
    train_data = dataset_prepare(train_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH, add_eos=True)
    print(f"Tokenize training dataset takes: {time.time() - start:.2f} seconds")

    # Create dataloaders
    print("\nCreating dataloaders...")
    # train_loader = get_dataloader(train_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=True)
    # valid_loader = get_dataloader(valid_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)
    # test_loader = get_dataloader(test_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = get_dataloader(train_data, vocab_zh, vocab_en, config.BATCH_SIZE, sampler=train_sampler)

    valid_sampler = DistributedSampler(valid_data, shuffle=False)
    valid_loader = get_dataloader(valid_data, vocab_zh, vocab_en, config.BATCH_SIZE, sampler=valid_sampler)
    
    test_sampler = DistributedSampler(test_data, shuffle=False)
    test_loader = get_dataloader(test_data, vocab_zh, vocab_en, config.BATCH_SIZE, sampler=test_sampler)

    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        d_model=config.TRANS_D_MODEL,
        num_heads=config.TRANS_NHEAD,
        num_encoder_layers=config.TRANS_NUM_ENCODER_LAYERS,
        num_decoder_layers=config.TRANS_NUM_DECODER_LAYERS,
        dim_feedforward=config.TRANS_DIM_FEEDFORWARD,
        dropout=config.TRANS_DROPOUT,
        position_embedding=config.POSITION_EMBEDDING,
        norm_type=config.NORM_TYPE,
        pad_idx=vocab_en.pad_idx
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    print_model_info(model, f"Transformer ({config.POSITION_EMBEDDING} pos, {config.NORM_TYPE})")

    # Loss and optimizer
    if args.label_smoothing:
        criterion = LabelSmoothingLoss(len(vocab_en), vocab_en.pad_idx, config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab_en.pad_idx)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001)

    # Learning rate scheduler with warmup
    scheduler = WarmupScheduler(optimizer, config.TRANS_D_MODEL, config.WARMUP_STEPS)

    # Training loop
    print("\nStarting training...")
    best_bleu = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")

        # Train
        train_loss, train_time = train_epoch(model, train_loader, optimizer, criterion, scheduler, config, epoch)
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")

        # Validate
        print("\nValidating...")
        valid_metrics, _, _ = evaluate(model, valid_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=False)

        print(f"Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"Valid BLEU-4: {valid_metrics['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Valid Precision-{n}: {valid_metrics[f'precision_{n}']:.4f}")

        # Save best model
        if valid_metrics['bleu4'] > best_bleu and local_rank == 0:
            best_bleu = valid_metrics['bleu4']
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path, vocab_zh, vocab_en)
            print(f"New best model saved! BLEU: {best_bleu:.4f}")

        # Save periodic checkpoint
        if epoch % 5 == 0 and local_rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path, vocab_zh, vocab_en)

    # Load best model
    if local_rank == 0:
        # Final evaluation on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)

        best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pt')
        load_checkpoint(best_checkpoint, model, device=device)

        # Test with greedy decoding
        print("\nGreedy Decoding:")
        test_metrics_greedy, refs, hyps_greedy = evaluate(model, test_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=False)
        print(f"Test BLEU-4: {test_metrics_greedy['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Test Precision-{n}: {test_metrics_greedy[f'precision_{n}']:.4f}")

        # Test with beam search
        print(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):")
        test_metrics_beam, _, hyps_beam = evaluate(model, test_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=True)
        print(f"Test BLEU-4: {test_metrics_beam['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Test Precision-{n}: {test_metrics_beam[f'precision_{n}']:.4f}")

        # Save results
        results_path = os.path.join(checkpoint_dir, 'results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"Transformer Configuration:\n")
            f.write(f"{'='*50}\n")
            f.write(f"Position Embedding: {config.POSITION_EMBEDDING}\n")
            f.write(f"Normalization: {config.NORM_TYPE}\n")
            f.write(f"d_model: {config.TRANS_D_MODEL}\n")
            f.write(f"num_layers: {config.TRANS_NUM_ENCODER_LAYERS}\n")
            f.write(f"num_heads: {config.TRANS_NHEAD}\n")
            f.write(f"batch_size: {config.BATCH_SIZE}\n")
            f.write(f"learning_rate: {config.LEARNING_RATE}\n")
            f.write(f"\nGreedy Decoding:\n")
            for metric, value in test_metrics_greedy.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):\n")
            for metric, value in test_metrics_beam.items():
                f.write(f"  {metric}: {value:.4f}\n")

            f.write("\n\nSample Translations (first 10):\n")
            f.write("="*50 + "\n")
            for i in range(min(10, len(refs))):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"Reference: {' '.join(refs[i])}\n")
                f.write(f"Greedy:    {' '.join(hyps_greedy[i])}\n")
                f.write(f"Beam:      {' '.join(hyps_beam[i])}\n")

        print(f"\nResults saved to {results_path}")
        print("Training complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer-based NMT")

    # Architectural ablations
    parser.add_argument('--position-embedding', type=str,
                       choices=['sinusoidal', 'learned', 'relative'],
                       help='Positional embedding type')
    parser.add_argument('--norm-type', type=str,
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Normalization type')

    # Hyperparameter sensitivity
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--d-model', type=int,
                       help='Model dimension')
    parser.add_argument('--num-layers', type=int,
                       help='Number of encoder/decoder layers')
    parser.add_argument('--num-heads', type=int,
                       help='Number of attention heads')

    # Training options
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large training set (100k)')
    parser.add_argument('--label-smoothing', action='store_true',
                       help='Use label smoothing')

    args = parser.parse_args()
    main(args)
