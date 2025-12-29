#!/usr/bin/env python3
"""
Generate visualization charts from experiment results.
Creates publication-ready figures for the project report.

Usage:
    python visualize_results.py --results ./results/experiment_results.json --output-dir ./figures
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server
matplotlib.use('Agg')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# Color palettes
COLORS = {
    'blue': '#2E86AB',
    'red': '#E94F37',
    'green': '#44AF69',
    'orange': '#F18F01',
    'purple': '#A23B72',
    'gray': '#6C757D',
}

PALETTE = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01', '#A23B72', '#6C757D']


def plot_rnn_attention_comparison(results, output_dir):
    """Plot RNN attention mechanism comparison."""
    data = results.get('rnn_attention', {})
    if not data:
        print("No RNN attention data found")
        return
    
    attention_types = list(data.keys())
    val_bleu = [data[a]['best_val_bleu'] for a in attention_types]
    test_bleu = [data[a]['final_test_bleu_beam'] for a in attention_types]
    
    x = np.arange(len(attention_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_bleu, width, label='Validation BLEU', color=COLORS['blue'])
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU (Beam)', color=COLORS['orange'])
    
    ax.set_xlabel('Attention Mechanism')
    ax.set_ylabel('BLEU-4 Score')
    ax.set_title('RNN Attention Mechanism Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attention_types])
    ax.legend()
    ax.set_ylim(0, max(max(val_bleu), max(test_bleu)) * 1.2)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'rnn_attention_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_rnn_teacher_forcing_comparison(results, output_dir):
    """Plot RNN teacher forcing strategy comparison."""
    data = results.get('rnn_teacher_forcing', {})
    if not data:
        print("No RNN teacher forcing data found")
        return
    
    tf_ratios = sorted(data.keys(), key=lambda x: float(x), reverse=True)
    val_bleu = [data[tf]['best_val_bleu'] for tf in tf_ratios]
    test_bleu = [data[tf]['final_test_bleu_beam'] for tf in tf_ratios]
    
    x = np.arange(len(tf_ratios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_bleu, width, label='Validation BLEU', color=COLORS['blue'])
    bars2 = ax.bar(x + width/2, test_bleu, width, label='Test BLEU (Beam)', color=COLORS['orange'])
    
    ax.set_xlabel('Teacher Forcing Ratio')
    ax.set_ylabel('BLEU-4 Score')
    ax.set_title('RNN Training Policy Comparison (Teacher Forcing)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'TF={tf}' for tf in tf_ratios])
    ax.legend()
    ax.set_ylim(0, max(max(val_bleu), max(test_bleu)) * 1.2)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'rnn_teacher_forcing_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_transformer_architecture_ablation(results, output_dir):
    """Plot Transformer architecture ablation (heatmap style)."""
    data = results.get('transformer_architecture', {})
    if not data:
        print("No Transformer architecture data found")
        return
    
    # Create matrix for heatmap
    pos_embeddings = ['sinusoidal', 'learned', 'relative']
    norm_types = ['LayerNorm', 'RMSNorm']
    
    matrix = np.zeros((len(pos_embeddings), len(norm_types)))
    for i, pos in enumerate(pos_embeddings):
        for j, norm in enumerate(norm_types):
            key = f"{pos}_{norm}"
            if key in data:
                matrix[i, j] = data[key]['best_val_bleu']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(norm_types)))
    ax.set_yticks(np.arange(len(pos_embeddings)))
    ax.set_xticklabels(norm_types)
    ax.set_yticklabels([p.capitalize() for p in pos_embeddings])
    
    ax.set_xlabel('Normalization Method')
    ax.set_ylabel('Position Embedding')
    ax.set_title('Transformer Architecture Ablation (Validation BLEU)')
    
    # Add text annotations
    for i in range(len(pos_embeddings)):
        for j in range(len(norm_types)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax, label='BLEU-4 Score')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'transformer_architecture_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")
    
    # Also create a bar chart version
    fig, ax = plt.subplots(figsize=(12, 6))
    configs = list(data.keys())
    bleu_scores = [data[c]['best_val_bleu'] for c in configs]
    
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(configs))]
    bars = ax.bar(configs, bleu_scores, color=colors)
    
    ax.set_xlabel('Configuration (Position_Normalization)')
    ax.set_ylabel('BLEU-4 Score')
    ax.set_title('Transformer Architecture Ablation')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'transformer_architecture_bar.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_hyperparameter_sensitivity(results, output_dir):
    """Plot Transformer hyperparameter sensitivity experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Batch Size
    data = results.get('transformer_batch_size', {})
    if data:
        ax = axes[0]
        batch_sizes = sorted(data.keys(), key=lambda x: int(x))
        bleu_scores = [data[bs]['best_val_bleu'] for bs in batch_sizes]
        ax.bar(batch_sizes, bleu_scores, color=COLORS['blue'])
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('BLEU-4 Score')
        ax.set_title('Batch Size Sensitivity')
        for i, (bs, bleu) in enumerate(zip(batch_sizes, bleu_scores)):
            ax.annotate(f'{bleu:.3f}', xy=(i, bleu), xytext=(0, 3),
                       textcoords="offset points", ha='center', fontsize=10)
    
    # Learning Rate
    data = results.get('transformer_learning_rate', {})
    if data:
        ax = axes[1]
        lrs = sorted(data.keys(), key=lambda x: float(x))
        bleu_scores = [data[lr]['best_val_bleu'] for lr in lrs]
        ax.bar(range(len(lrs)), bleu_scores, color=COLORS['green'])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('BLEU-4 Score')
        ax.set_title('Learning Rate Sensitivity')
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels(lrs, rotation=45)
        for i, bleu in enumerate(bleu_scores):
            ax.annotate(f'{bleu:.3f}', xy=(i, bleu), xytext=(0, 3),
                       textcoords="offset points", ha='center', fontsize=10)
    
    # Model Scale
    data = results.get('transformer_model_scale', {})
    if data:
        ax = axes[2]
        scales = ['small', 'medium', 'large']
        scales = [s for s in scales if s in data]
        bleu_scores = [data[s]['best_val_bleu'] for s in scales]
        ax.bar(scales, bleu_scores, color=COLORS['orange'])
        ax.set_xlabel('Model Scale')
        ax.set_ylabel('BLEU-4 Score')
        ax.set_title('Model Scale Sensitivity')
        for i, bleu in enumerate(bleu_scores):
            ax.annotate(f'{bleu:.3f}', xy=(i, bleu), xytext=(0, 3),
                       textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'transformer_hyperparameter_sensitivity.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_rnn_vs_transformer_comparison(results, output_dir):
    """Plot comparison between best RNN and best Transformer models."""
    # Get best RNN result
    rnn_data = results.get('rnn_attention', {})
    best_rnn_bleu = 0
    best_rnn_name = ""
    for name, metrics in rnn_data.items():
        if metrics['final_test_bleu_beam'] > best_rnn_bleu:
            best_rnn_bleu = metrics['final_test_bleu_beam']
            best_rnn_name = f"RNN-{name}"
    
    # Get best Transformer result
    trans_data = results.get('transformer_architecture', {})
    best_trans_bleu = 0
    best_trans_name = ""
    for name, metrics in trans_data.items():
        if metrics['final_test_bleu_beam'] > best_trans_bleu:
            best_trans_bleu = metrics['final_test_bleu_beam']
            best_trans_name = f"Transformer-{name}"
    
    # Get finetune result
    finetune_data = results.get('finetune', {})
    best_ft_bleu = 0
    best_ft_name = ""
    for name, metrics in finetune_data.items():
        if metrics.get('best_val_bleu', 0) > best_ft_bleu:
            best_ft_bleu = metrics.get('best_val_bleu', 0)
            best_ft_name = "mT5-Finetuned"
    
    # Create comparison chart
    models = [best_rnn_name, best_trans_name]
    scores = [best_rnn_bleu, best_trans_bleu]
    
    if best_ft_bleu > 0:
        models.append(best_ft_name)
        scores.append(best_ft_bleu)
    
    if not any(scores):
        print("No comparison data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green']][:len(models)]
    bars = ax.bar(models, scores, color=colors)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('BLEU-4 Score (Test, Beam Search)')
    ax.set_title('RNN vs Transformer vs Finetuned Model Comparison')
    ax.set_ylim(0, max(scores) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_greedy_vs_beam_comparison(results, output_dir):
    """Plot Greedy vs Beam Search decoding comparison."""
    # Collect all models with both greedy and beam results
    all_data = {}
    
    for category in ['rnn_attention', 'transformer_architecture']:
        for name, metrics in results.get(category, {}).items():
            if metrics['final_test_bleu_greedy'] > 0 or metrics['final_test_bleu_beam'] > 0:
                key = f"{category.split('_')[0]}-{name}"
                all_data[key] = {
                    'greedy': metrics['final_test_bleu_greedy'],
                    'beam': metrics['final_test_bleu_beam']
                }
    
    if not all_data:
        print("No decoding comparison data available")
        return
    
    models = list(all_data.keys())[:6]  # Limit to 6 for readability
    greedy = [all_data[m]['greedy'] for m in models]
    beam = [all_data[m]['beam'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, greedy, width, label='Greedy Decoding', color=COLORS['blue'])
    bars2 = ax.bar(x + width/2, beam, width, label='Beam Search', color=COLORS['orange'])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('BLEU-4 Score')
    ax.set_title('Greedy vs Beam Search Decoding Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'decoding_strategy_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization charts")
    parser.add_argument('--results', type=str, default='./results/experiment_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='./figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        print("Please run extract_results.py first.")
        return
    
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n📊 Generating Visualizations...")
    print("=" * 50)
    
    # Generate all plots
    plot_rnn_attention_comparison(results, args.output_dir)
    plot_rnn_teacher_forcing_comparison(results, args.output_dir)
    plot_transformer_architecture_ablation(results, args.output_dir)
    plot_hyperparameter_sensitivity(results, args.output_dir)
    plot_rnn_vs_transformer_comparison(results, args.output_dir)
    plot_greedy_vs_beam_comparison(results, args.output_dir)
    
    print("\n" + "=" * 50)
    print(f"✅ All figures saved to: {args.output_dir}")
    print("\nGenerated figures:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
