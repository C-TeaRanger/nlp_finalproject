#!/usr/bin/env python3
"""
Extract experiment results from training logs.
Parses log files and outputs a structured JSON with all BLEU scores.

Usage:
    python extract_results.py --log-dir ./logs --output results.json
"""

import os
import re
import json
import argparse
from collections import defaultdict


def extract_metrics_from_log(log_path):
    """
    Extract metrics from a single log file.
    Returns the best validation BLEU and final test BLEU scores.
    """
    results = {
        'best_val_bleu': 0.0,
        'final_test_bleu_greedy': 0.0,
        'final_test_bleu_beam': 0.0,
        'final_train_loss': 0.0,
        'final_val_loss': 0.0,
        'epochs_completed': 0,
        'precision_1': 0.0,
        'precision_2': 0.0,
        'precision_3': 0.0,
        'precision_4': 0.0,
    }
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return results
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract best validation BLEU
    val_bleu_matches = re.findall(r'Valid BLEU-4:\s*([\d.]+)', content)
    if val_bleu_matches:
        results['best_val_bleu'] = max(float(b) for b in val_bleu_matches)
    
    # Extract test BLEU (greedy)
    greedy_section = re.search(r'Greedy Decoding:.*?Test BLEU-4:\s*([\d.]+)', content, re.DOTALL)
    if greedy_section:
        results['final_test_bleu_greedy'] = float(greedy_section.group(1))
    
    # Extract test BLEU (beam search)
    beam_section = re.search(r'Beam Search Decoding.*?Test BLEU-4:\s*([\d.]+)', content, re.DOTALL)
    if beam_section:
        results['final_test_bleu_beam'] = float(beam_section.group(1))
    
    # Extract train loss
    train_loss_matches = re.findall(r'Train Loss:\s*([\d.]+)', content)
    if train_loss_matches:
        results['final_train_loss'] = float(train_loss_matches[-1])
    
    # Extract validation loss
    val_loss_matches = re.findall(r'Valid Loss:\s*([\d.]+)', content)
    if val_loss_matches:
        results['final_val_loss'] = float(val_loss_matches[-1])
    
    # Extract epochs completed
    epoch_matches = re.findall(r'Epoch\s+(\d+)/', content)
    if epoch_matches:
        results['epochs_completed'] = max(int(e) for e in epoch_matches)
    
    # Extract precision scores (from test evaluation)
    for n in range(1, 5):
        precision_matches = re.findall(rf'Test Precision-{n}:\s*([\d.]+)', content)
        if precision_matches:
            results[f'precision_{n}'] = float(precision_matches[-1])
    
    # Alternative patterns for precision
    if results['precision_1'] == 0:
        for n in range(1, 5):
            alt_matches = re.findall(rf'precision_{n}:\s*([\d.]+)', content)
            if alt_matches:
                results[f'precision_{n}'] = float(alt_matches[-1])
    
    return results


def parse_all_logs(log_dir):
    """
    Parse all log files and organize results by experiment type.
    """
    all_results = {
        'rnn_attention': {},           # dot, multiplicative, additive
        'rnn_teacher_forcing': {},     # 1.0, 0.5, 0.0
        'transformer_architecture': {},  # position x normalization
        'transformer_batch_size': {},    # 32, 64, 128
        'transformer_learning_rate': {}, # different lr values
        'transformer_model_scale': {},   # small, medium, large
        'finetune': {},                  # mT5
    }
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return all_results
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    print(f"Found {len(log_files)} log files")
    
    for log_file in sorted(log_files):
        log_path = os.path.join(log_dir, log_file)
        metrics = extract_metrics_from_log(log_path)
        
        # Categorize by experiment type
        name = log_file.replace('.log', '')
        
        # RNN attention experiments
        if name in ['rnn_dot', 'rnn_multiplicative', 'rnn_additive']:
            attn_type = name.replace('rnn_', '')
            all_results['rnn_attention'][attn_type] = metrics
            # rnn_additive with TF=1.0 is also the baseline for teacher forcing comparison
            if name == 'rnn_additive':
                all_results['rnn_teacher_forcing']['1.0'] = metrics
            
        # RNN teacher forcing experiments
        elif 'rnn_tf_' in name:
            tf_ratio = name.replace('rnn_tf_', '')
            all_results['rnn_teacher_forcing'][tf_ratio] = metrics
            
        # Transformer architecture experiments
        elif name.startswith('transformer_') and ('sinusoidal' in name or 'learned' in name or 'relative' in name):
            # e.g., transformer_sinusoidal_LayerNorm
            parts = name.replace('transformer_', '').split('_')
            if len(parts) >= 2:
                pos_emb = parts[0]
                norm = parts[1]
                key = f"{pos_emb}_{norm}"
                all_results['transformer_architecture'][key] = metrics
                
        # Transformer batch size experiments
        elif 'transformer_bs_' in name:
            bs = name.replace('transformer_bs_', '')
            all_results['transformer_batch_size'][bs] = metrics
            
        # Transformer learning rate experiments
        elif 'transformer_lr_' in name:
            lr = name.replace('transformer_lr_', '')
            all_results['transformer_learning_rate'][lr] = metrics
            
        # Transformer model scale experiments
        elif name in ['transformer_small', 'transformer_medium', 'transformer_large']:
            scale = name.replace('transformer_', '')
            all_results['transformer_model_scale'][scale] = metrics
            
        # Finetuning experiments
        elif 'finetune' in name or 'mt5' in name.lower():
            all_results['finetune'][name] = metrics
    
    return all_results


def print_summary(results):
    """Print a formatted summary of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # RNN Attention Comparison
    print("\n📊 RNN Attention Mechanism Comparison:")
    print("-" * 50)
    print(f"{'Attention Type':<20} {'Val BLEU':<12} {'Test BLEU (Beam)':<15}")
    for attn, metrics in sorted(results['rnn_attention'].items()):
        print(f"{attn:<20} {metrics['best_val_bleu']:<12.4f} {metrics['final_test_bleu_beam']:<15.4f}")
    
    # RNN Teacher Forcing Comparison
    print("\n📊 RNN Training Policy Comparison (Teacher Forcing):")
    print("-" * 50)
    print(f"{'TF Ratio':<20} {'Val BLEU':<12} {'Test BLEU (Beam)':<15}")
    for tf, metrics in sorted(results['rnn_teacher_forcing'].items()):
        print(f"{tf:<20} {metrics['best_val_bleu']:<12.4f} {metrics['final_test_bleu_beam']:<15.4f}")
    
    # Transformer Architecture Ablation
    print("\n📊 Transformer Architecture Ablation:")
    print("-" * 50)
    print(f"{'Configuration':<25} {'Val BLEU':<12} {'Test BLEU (Beam)':<15}")
    for config, metrics in sorted(results['transformer_architecture'].items()):
        print(f"{config:<25} {metrics['best_val_bleu']:<12.4f} {metrics['final_test_bleu_beam']:<15.4f}")
    
    # Transformer Hyperparameter Sensitivity
    print("\n📊 Transformer Batch Size Sensitivity:")
    print("-" * 50)
    for bs, metrics in sorted(results['transformer_batch_size'].items()):
        print(f"BS={bs:<10} Val BLEU: {metrics['best_val_bleu']:.4f}")
    
    print("\n📊 Transformer Learning Rate Sensitivity:")
    print("-" * 50)
    for lr, metrics in sorted(results['transformer_learning_rate'].items()):
        print(f"LR={lr:<10} Val BLEU: {metrics['best_val_bleu']:.4f}")
    
    print("\n📊 Transformer Model Scale:")
    print("-" * 50)
    for scale, metrics in sorted(results['transformer_model_scale'].items()):
        print(f"{scale:<10} Val BLEU: {metrics['best_val_bleu']:.4f}")
    
    # Finetuning Results
    if results['finetune']:
        print("\n📊 Pretrained Model Finetuning:")
        print("-" * 50)
        for name, metrics in results['finetune'].items():
            print(f"{name}: Val BLEU: {metrics['best_val_bleu']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Extract experiment results from logs")
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory containing log files')
    parser.add_argument('--output', type=str, default='./results/experiment_results.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Parse all logs
    results = parse_all_logs(args.log_dir)
    
    # Print summary
    print_summary(results)
    
    # Save to JSON
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
