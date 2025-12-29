"""
Training script for RNN-based machine translation
Supports different attention mechanisms and teacher forcing strategies
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from datetime import datetime
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
from models.rnn.seq2seq import Seq2Seq
from utils.metrics import calculate_all_metrics
from utils.training_utils import (
    save_checkpoint, load_checkpoint, count_parameters,
    AverageMeter, Timer, print_model_info, get_lr
)
from utils.beam_search import greedy_decode, beam_search_decode
from data.vocabulary_bpe_en import BPEVocabularyEN
from data.vocabulary_bpe_zh import BPEVocabularyZH
# from data.vocabulary_hanlp import HanLPVocabulary
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_unused_parameters(model):
    unused_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            unused_params.append(name)
    if unused_params:
        print("Unused parameters found:")
        for name in unused_params:
            print(f"  - {name}")
    return unused_params


def train_epoch(model, dataloader, optimizer, criterion, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    timer = Timer()
    timer.start()

    current_ratio = config.TEACHER_FORCING_RATIO
    # current_ratio = max(config.MIN_RATIO, config.TEACHER_FORCING_RATIO - (config.DECAY_RATE * (epoch - 1)))
    print(f"Current teacher forcing ratio: {current_ratio:.2f}")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(pbar):
        # breakpoint()

        # Move to device
        src = src.to(config.DEVICE)
        tgt = tgt.to(config.DEVICE)
        src_len = src_len.to(config.DEVICE)

        # Forward pass
        outputs, _ = model(src, src_len, tgt, teacher_forcing_ratio=current_ratio)
        # outputs: (batch_size, tgt_len, vocab_size)'
        # Compute loss (ignore first token which is <bos>)
        output_dim = outputs.size(-1)
        outputs = outputs[:, 1:, :].reshape(-1, output_dim)  # (batch_size * (tgt_len-1), vocab_size)
        # outputs = outputs[:, :-1, :].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)  # (batch_size * (tgt_len-1)), no <bos>

        loss = criterion(outputs, tgt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        # find_unused_parameters(model)

        optimizer.step()

        # # Update learning rate
        # if scheduler is not None:
        #     scheduler.step()

        # Update metrics
        losses.update(loss.item(), src.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
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
            outputs, _ = model(src, src_len, tgt, teacher_forcing_ratio=0.0)

            # Compute loss
            output_dim = outputs.size(-1)
            outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
            # outputs_flat = outputs[:, :-1, :].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, tgt_flat)
            losses.update(loss.item(), src.size(0))

            # breakpoint()

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
                # ref_words = vocab_tgt.ids_to_tokens(ref_tokens, skip_special=False)
                # pred_words = vocab_tgt.ids_to_tokens(pred_tokens, skip_special=False)
                ref_words = vocab_tgt.decode(ref_tokens, skip_special=True).split()
                pred_words = vocab_tgt.decode(pred_tokens, skip_special=True).split()
                # ref_words = vocab_tgt.ids_to_tokens(ref_tokens, skip_special=True)
                # pred_words = vocab_tgt.ids_to_tokens(pred_tokens, skip_special=True)

                # breakpoint()

                references.append(ref_words)
                hypotheses.append(pred_words)

    # Calculate metrics
    metrics = calculate_all_metrics(references, hypotheses)
    metrics['loss'] = losses.avg

    # breakpoint()

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

    if local_rank == 0:
        print(f"Using device: {device}")

    config = Config()

    # Override config with command line arguments
    if args.attention:
        config.ATTENTION_TYPE = args.attention
    if args.teacher_forcing or args.teacher_forcing == 0.0:
        config.TEACHER_FORCING_RATIO = args.teacher_forcing
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    config.DEVICE = device
    # # Set device
    # device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    # config.DEVICE = device
    # print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = config.RNN_CHECKPOINT_DIR
    exp_name = f"{config.ATTENTION_TYPE}_LSTM_tf{config.TEACHER_FORCING_RATIO}_bs{config.BATCH_SIZE}_lr{config.LEARNING_RATE}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = Preprocessor()

    train_path = config.TRAIN_LARGE_PATH if args.use_large else config.TRAIN_SMALL_PATH
    train_data = preprocessor.load_and_preprocess(train_path, return_text=True, zh_key="zh_hy")
    valid_data = preprocessor.load_and_preprocess(config.VALID_PATH, return_text=True, zh_key="zh_hy")
    test_data = preprocessor.load_and_preprocess(config.TEST_PATH, return_text=True, zh_key="zh_hy")

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build vocabularies
    print("\nBuilding vocabularies...")
    zh_sentences = [zh for zh, en in train_data]
    en_sentences = [en for zh, en in train_data]

    # # using BPE
    vocab_zh = BPEVocabularyZH("Chinese")
    vocab_en = BPEVocabularyEN("English")

    if local_rank == 0:
        vocab_zh.train_from_texts(zh_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
        vocab_zh.save(os.path.join(checkpoint_dir, 'tokenizer_zh.pkl'), os.path.join(checkpoint_dir, 'vocab_zh.json'))
        vocab_en.train_from_texts(en_sentences, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)
        vocab_en.save(os.path.join(checkpoint_dir, 'tokenizer_en.pkl'), os.path.join(checkpoint_dir, 'vocab_en.json'))

    # Synchronize all processes
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Add delay for NFS caching
    time.sleep(10)

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

    print("Vocabulary loaded.")

    print("\nPreparing datasets...")

    start = time.time()
    test_data = dataset_prepare(test_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH)
    print(f"Tokenize testing dataset takes: {time.time() - start:.2f} seconds")
    # time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # save_file_name = f"./dataset/test_data_{time_stamp}.json"
    # with open(save_file_name, 'w', encoding='utf-8') as f:
    #     json.dump(test_data, f, ensure_ascii=False, indent=4)
    # print(f"Testing dataset has been saved to {save_file_name}.")

    start = time.time()
    valid_data = dataset_prepare(valid_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH)
    print(f"Tokenize validation dataset takes: {time.time() - start:.2f} seconds")
    # time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # save_file_name = f"./dataset/valid_data_{time_stamp}.json"
    # with open(save_file_name, 'w', encoding='utf-8') as f:
    #     json.dump(valid_data, f, ensure_ascii=False, indent=4)
    # print(f"Validation dataset has been saved to {save_file_name}.")

    start = time.time()
    train_data = dataset_prepare(train_data, vocab_zh, vocab_en, max_length=config.MAX_LENGTH, add_eos=True)
    print(f"Tokenize training dataset takes: {time.time() - start:.2f} seconds")
    # time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # save_file_name = f"./dataset/training_data_10k_{time_stamp}.json"
    # with open(save_file_name, 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
    # print(f"Traning dataset has been saved to {save_file_name}.")

    # training_data_file = "./dataset/training_data_10k_20251219_1823.json"
    # with open(training_data_file, 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    # print(f"Traning dataset has been loaded from {training_data_file}.")

    # valid_data_file = "./dataset/valid_data_20251219_1823.json"
    # with open(valid_data_file, 'r', encoding='utf-8') as f:
    #     valid_data = json.load(f)
    # print(f"Validation dataset has been loaded from {valid_data_file}.")

    # test_data_file = "./dataset/test_data_20251219_1823.json"
    # with open(test_data_file, 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)
    # print(f"Testing dataset has been loaded from {test_data_file}.")

    # breakpoint()
    # train_data = [(vocab_zh.encode(zh), vocab_en.encode(en)) for zh, en in train_data]
    # valid_data = [(vocab_zh.encode(zh), vocab_en.encode(en)) for zh, en in valid_data]
    # test_data = [(vocab_zh.encode(zh), vocab_en.encode(en)) for zh, en in test_data]

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
    model = Seq2Seq(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        embed_dim=config.RNN_EMBED_DIM,
        hidden_dim=config.RNN_HIDDEN_DIM,
        num_layers=config.RNN_NUM_LAYERS,
        dropout=config.RNN_DROPOUT,
        cell_type=config.RNN_CELL_TYPE,
        attention_type=config.ATTENTION_TYPE,
        pad_idx=vocab_en.pad_idx
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    print_model_info(model, f"RNN-{config.RNN_CELL_TYPE} ({config.ATTENTION_TYPE} attention)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_en.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # 我们希望 Loss 越小越好
        factor=0.5,      # 每次衰减的比例。建议设为 0.5 (温和) 或 0.1 (激进)
        patience=3,      # 容忍度。如果 Val Loss 连续 3 个 Epoch 不下降，就触发衰减
        verbose=True,    # 打印日志，让你知道什么时候 LR 变了
        min_lr=1e-6      # 学习率下限，防止变成 0
    )  # not used

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
        # valid_metrics, _, _ = evaluate(model, train_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=False)

        print(f"Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"Valid BLEU-4: {valid_metrics['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Valid Precision-{n}: {valid_metrics[f'precision_{n}']:.4f}")

        # if scheduler is not None:
        #     scheduler.step(valid_metrics['loss'])

        # Save best model
        if valid_metrics['bleu4'] > best_bleu:
            best_bleu = valid_metrics['bleu4']
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path)
            print(f"New best model saved! BLEU: {best_bleu:.4f}")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path)

    if local_rank == 0:
        # Final evaluation on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)

        # Load best model
        best_checkpoint = os.path.join(checkpoint_dir, f'best_model_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
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
        results_path = os.path.join(checkpoint_dir, f'results_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"RNN-{config.RNN_CELL_TYPE} with {config.ATTENTION_TYPE} attention\n")
            f.write(f"{'='*50}\n\n")
            f.write("Greedy Decoding:\n")
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
    parser = argparse.ArgumentParser(description="Train RNN-based NMT")
    parser.add_argument('--attention', type=str, choices=['dot', 'multiplicative', 'additive'],
                       help='Attention mechanism')
    # parser.add_argument('--cell', type=str, choices=['LSTM', 'GRU'],
    #                    help='RNN cell type')
    parser.add_argument('--teacher-forcing', type=float,
                       help='Teacher forcing ratio (0.0-1.0)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large training set (100k)')

    args = parser.parse_args()
    main(args)
