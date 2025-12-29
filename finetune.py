"""
Finetuning script for pretrained Transformer models (T5, mT5, etc.) on NMT tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.preprocessor import Preprocessor
# from data.vocabulary import Vocabulary
from data.vocabulary_bpe_en import BPEVocabularyEN
from data.vocabulary_bpe_zh import BPEVocabularyZH
from data.dataloader import get_dataloader
from utils.metrics import calculate_all_metrics
from utils.training_utils import (
    save_checkpoint, load_checkpoint, count_parameters,
    AverageMeter, Timer, print_model_info, WarmupScheduler
)
from utils.beam_search import greedy_decode, beam_search_decode
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(self.data[idx])
        return self.data[idx][0], self.data[idx][1]


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, tokenizer):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    timer = Timer()
    timer.start()
    prefix = "translate Chinese to English: "

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    # for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(pbar):
    for batch_idx, (src_texts, tgt_texts) in enumerate(pbar):
        src_inputs = [prefix + text for text in src_texts]
        # breakpoint()

        # Decode token IDs back to text
        # src_texts = [tokenizer.decode(s, skip_special_tokens=True) for s in src]
        # tgt_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in tgt]

        model_inputs = tokenizer(src_inputs, max_length=128, truncation=True, padding=True, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt_texts, max_length=128, truncation=True, padding=True, return_tensors="pt")

        input_ids = model_inputs.input_ids.to(config.DEVICE)
        attention_mask = model_inputs.attention_mask.to(config.DEVICE)
        labels = labels.input_ids.to(config.DEVICE)
        labels[labels == tokenizer.pad_token_id] = -100

        # # Tokenize with pretrained tokenizer
        # src_encoded = tokenizer(src_texts, padding=True, truncation=True,
        #                        max_length=config.MAX_LEN, return_tensors='pt')
        # tgt_encoded = tokenizer(tgt_texts, padding=True, truncation=True,
        #                        max_length=config.MAX_LEN, return_tensors='pt')

        # # Move to device
        # input_ids = src_encoded['input_ids'].to(config.DEVICE)
        # attention_mask = src_encoded['attention_mask'].to(config.DEVICE)
        # labels = tgt_encoded['input_ids'].to(config.DEVICE)
        # labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        # Update metrics
        losses.update(loss.item(), input_ids.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    elapsed = timer.stop()
    return losses.avg, elapsed


def evaluate(model, dataloader, config, tokenizer, device):
    """Evaluate on validation/test set"""
    model.eval()
    losses = AverageMeter()
    prefix = "translate Chinese to English: "

    hypotheses = []
    references = []

    with torch.no_grad():
        for src_texts, tgt_texts in tqdm(dataloader, desc="Evaluating"):
            src_inputs = [prefix + text for text in src_texts]
            
            model_inputs = tokenizer(src_inputs, max_length=128, truncation=True, padding=True, return_tensors="pt")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(tgt_texts, max_length=128, truncation=True, padding=True, return_tensors="pt")

            input_ids = model_inputs.input_ids.to(device)
            attention_mask = model_inputs.attention_mask.to(device)
            labels = labels.input_ids.to(device)
            
            # Loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.update(outputs.loss.item(), input_ids.size(0))

            # Generate translations
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=config.BEAM_SIZE,
                length_penalty=0.6,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

            # Decode predictions and references
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for i, pred in enumerate(preds):
                preds[i] = pred.split(" ")

            for i, ref in enumerate(refs):
                refs[i] = ref.split(" ")

            # breakpoint()

            hypotheses.extend(preds)
            references.extend(refs)

        # Calculate metrics
        metrics = calculate_all_metrics(hypotheses, references)
        metrics['loss'] = losses.avg

    return metrics, references, hypotheses


def main():
    parser = argparse.ArgumentParser(description='Finetune pretrained model for NMT')
    parser.add_argument('--model-name', type=str, default='google/mt5-small',
                       help='Pretrained model name (google/mt5-small, google/mt5-base, google/mt5-large, etc.)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,  # 5
                       help='Number of epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large dataset (100k)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder parameters')

    args = parser.parse_args()

    # Setup
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.EPOCHS = args.epochs
    config.WARMUP_STEPS = args.warmup_steps

    os.makedirs('./checkpoints/pretrained', exist_ok=True)

    model_name_safe = args.model_name.replace('/', '_')
    local_model_path = os.path.join("./pretrained_models", model_name_safe)

    print(f"Loading pretrained model: {args.model_name}")

    if os.path.exists(local_model_path):
        print(f"Found local model at {local_model_path}, loading...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    else:
        print(f"Local model not found. Downloading {args.model_name} from Hub...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        
        print(f"Saving model to {local_model_path}...")
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

    # # Load tokenizer and model
    # print(f"Loading pretrained model: {args.model_name}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(config.DEVICE)

    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")

    print(f"Model parameters: {count_parameters(model):,}")

    # Load data
    print("Loading data...")

    preprocessor = Preprocessor()
    train_path = config.TRAIN_LARGE_PATH if args.use_large else config.TRAIN_SMALL_PATH
    raw_train_data = preprocessor.load_and_preprocess(train_path, return_text=True)
    raw_valid_data = preprocessor.load_and_preprocess(config.VALID_PATH, return_text=True)
    raw_test_data = preprocessor.load_and_preprocess(config.TEST_PATH, return_text=True)

    train_dataset = TranslationDataset(raw_train_data)
    valid_dataset = TranslationDataset(raw_valid_data)
    test_dataset = TranslationDataset(raw_test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # vocab_src = BPEVocabularyZH("Chinese")
    # vocab_tgt = BPEVocabularyEN("English")

    # dataset_type = 'train_100k' if args.use_large else 'train_10k'
    # train_dataloader = get_dataloader(
    #     train_data, vocab_src, vocab_tgt, config.BATCH_SIZE, preprocessor
    # )
    # valid_dataloader = get_dataloader(
    #     valid_data, vocab_src, vocab_tgt, config.BATCH_SIZE, preprocessor
    # )
    # test_dataloader = get_dataloader(
    #     test_data, vocab_src, vocab_tgt, config.BATCH_SIZE, preprocessor
    # )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    total_steps = len(train_dataloader) * config.EPOCHS
    # scheduler = WarmupScheduler(optimizer, config.WARMUP_STEPS, total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS, 
        num_training_steps=total_steps
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Training loop
    best_bleu = 0
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, train_time = train_epoch(
            model, train_dataloader, optimizer, scheduler, config, epoch + 1, tokenizer
        )
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")

        # Validate
        valid_metrics, _, _ = evaluate(
            model, valid_dataloader, config, tokenizer, config.DEVICE
        )
        print(f"Valid Loss: {valid_metrics['loss']:.4f} | BLEU: {valid_metrics['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Valid Precision-{n}: {valid_metrics[f'precision_{n}']:.4f}")

        # Save checkpoint
        checkpoint_path = f'./checkpoints/pretrained/{args.model_name.split("/")[1]}_epoch_{epoch + 1}.pt'
        # save_checkpoint({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, checkpoint_path)
        # os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path)

        # Save best model
        if valid_metrics['bleu4'] > best_bleu:
            best_bleu = valid_metrics['bleu4'] 
            best_path = f'./checkpoints/pretrained/{args.model_name.split("/")[1]}_best.pt'
            # save_checkpoint({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, best_path)
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], best_path)
            print(f"Best model saved with BLEU: {best_bleu:.4f}")

    load_checkpoint(best_path, model, device=config.DEVICE)

    # # Test with greedy decoding
    # print("\nGreedy Decoding:")
    # test_metrics_greedy, refs, hyps_greedy = evaluate(model, test_loader, config, tokenizer, config.DEVICE)
    # print(f"Test BLEU-4: {test_metrics_greedy['bleu4']:.4f}")
    # for n in range(1, 5):
    #     print(f"Test Precision-{n}: {test_metrics_greedy[f'precision_{n}']:.4f}")

    # Test with beam search
    print(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):")
    test_metrics_beam, _, hyps_beam = evaluate(model, test_dataloader, config, tokenizer, config.DEVICE)
    print(f"Test BLEU-4: {test_metrics_beam['bleu4']:.4f}")
    for n in range(1, 5):
        print(f"Test Precision-{n}: {test_metrics_beam[f'precision_{n}']:.4f}")

    # # Test
    # print(f"\n{'='*60}")
    # print("Testing on test set...")
    # print(f"{'='*60}")
    # test_loss, test_bleu, test_precision = evaluate(
    #     model, test_dataloader, config, tokenizer, vocab_tgt
    # )
    # print(f"Test Loss: {test_loss:.4f} | BLEU: {test_bleu:.4f}")
    # print(f"Precision-1: {test_precision[0]:.4f} | Precision-4: {test_precision[3]:.4f}")


def test_main():
    parser = argparse.ArgumentParser(description='Finetune pretrained model for NMT')
    parser.add_argument('--model-name', type=str, default='google/mt5-large',
                       help='Pretrained model name (google/mt5-small, google/mt5-base, google/mt5-large, etc.)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,  # 5
                       help='Number of epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large dataset (100k)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder parameters')

    args = parser.parse_args()

    # Setup
    config = Config()

    os.makedirs('./checkpoints/pretrained', exist_ok=True)

    model_name_safe = args.model_name.replace('/', '_')
    local_model_path = os.path.join("./pretrained_models", model_name_safe)

    if os.path.exists(local_model_path):
        print(f"Found local model at {local_model_path}, loading...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    else:
        print(f"Local model not found. Downloading {args.model_name} from Hub...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        
        print(f"Saving model to {local_model_path}...")
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

    # Load data
    print("Loading data...")

    preprocessor = Preprocessor()
    raw_test_data = preprocessor.load_and_preprocess(config.TEST_PATH, return_text=True)
    test_dataset = TranslationDataset(raw_test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    load_checkpoint(args.checkpoint, model, device=config.DEVICE)
    print(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):")
    test_metrics_beam, _, hyps_beam = evaluate(model, test_dataloader, config, tokenizer, config.DEVICE)
    print(f"Test BLEU-4: {test_metrics_beam['bleu4']:.4f}")
    for n in range(1, 5):
        print(f"Test Precision-{n}: {test_metrics_beam[f'precision_{n}']:.4f}")


if __name__ == '__main__':
    main()
