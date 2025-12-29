"""
Training utilities
"""

import torch
import os
import time
from typing import Optional


def save_checkpoint(model, optimizer, epoch: int, batch_idx: int, loss: float,
                   filepath: str, vocab_src=None, vocab_tgt=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save vocabularies if provided
    if vocab_src is not None:
        checkpoint['vocab_src'] = vocab_src
    if vocab_tgt is not None:
        checkpoint['vocab_tgt'] = vocab_tgt

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model=None, optimizer=None, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)

    # model.load_state_dict(checkpoint['model_state_dict'])
    # if optimizer is not None and 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    loss = checkpoint.get('loss', 0.0)

    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}")

    return checkpoint


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def reset(self):
        self.start_time = None
        self.elapsed = 0


def create_masks(src, tgt, pad_idx):
    """
    Create masks for Transformer

    Args:
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        pad_idx: Padding index

    Returns:
        src_mask: (batch_size, 1, src_len)
        tgt_mask: (batch_size, tgt_len, tgt_len)
        src_padding_mask: (batch_size, src_len)
        tgt_padding_mask: (batch_size, tgt_len)
    """
    # Source padding mask
    src_padding_mask = (src == pad_idx)  # (batch_size, src_len)

    # Target padding mask
    tgt_padding_mask = (tgt == pad_idx)  # (batch_size, tgt_len)

    # Target attention mask (causal mask)
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    # (tgt_len, tgt_len)

    return src_padding_mask, tgt_padding_mask, tgt_mask


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class WarmupScheduler:
    """Learning rate scheduler with warmup"""

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        base_lr = self.optimizer.defaults['lr']
        return base_lr * self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        ) * self.warmup_steps


def print_model_info(model, model_name: str):
    """Print model information"""
    print(f"\n{'='*50}")
    print(f"{model_name} Model Information")
    print(f"{'='*50}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model architecture:\n{model}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Training utilities implemented")
