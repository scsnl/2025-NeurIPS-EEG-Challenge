#!/usr/bin/env python3
"""
Plot training and validation loss from metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np
import glob
import torch

def plot_losses(metrics_dir):
    """Plot train and val losses from metrics.csv"""
    metrics_dir = Path(metrics_dir)
    csv_file = metrics_dir / "metrics.csv"
    
    if not csv_file.exists():
        print(f"Error: metrics.csv not found in {metrics_dir}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract train and val losses (handle empty cells)
    train_losses = df[df['loss'].notna() & (df['loss'] != '')][['epoch', 'loss']].copy()
    val_losses = df[df['val_loss'].notna() & (df['val_loss'] != '')][['epoch', 'val_loss']].copy()
    
    # Extract test loss if available
    test_loss_value = None
    if 'test_loss' in df.columns:
        test_losses = df[df['test_loss'].notna() & (df['test_loss'] != '')]['test_loss'].copy()
        if not test_losses.empty:
            test_loss_value = pd.to_numeric(test_losses.iloc[0], errors='coerce')
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses['epoch'], train_losses['loss'], 'b-', label='Train Loss', linewidth=2)
    if test_loss_value is not None and not pd.isna(test_loss_value):
        ax1.axhline(y=test_loss_value, color='green', linestyle='--', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_ylim(0.0, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation loss
    ax2.plot(val_losses['epoch'], val_losses['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if test_loss_value is not None and not pd.isna(test_loss_value):
        ax2.axhline(y=test_loss_value, color='green', linestyle='--', label='Test Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.set_ylim(0.0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create plots directory at same level as csv directory
    #plots_dir = (metrics_dir.parent).parent / "plots"
    #plots_dir.mkdir(exist_ok=True)
    
    # Save the plot
    plot_file = metrics_dir / "loss_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {plot_file}")

def plot_model_comparison(base_dir):
    """Plot minimum loss comparison across models with error bars."""
    base_dir = Path(base_dir)
    
    # Storage for metrics
    model_names = []
    train_loss_min = []
    val_loss_min = []
    train_loss_std = []
    val_loss_std = []
    test_loss_values = []
    test_nrmse_values = []
    
    # Find all stats_* directories
    stats_dirs = list(base_dir.glob("stats_*"))
    
    if not stats_dirs:
        print(f"No stats_* directories found in {base_dir}")
        return
    
    print(f"Found {len(stats_dirs)} model directories")
    
    # Process each model
    for stats_dir in stats_dirs:
        model_name = stats_dir.name.replace("stats_", "")
        print(f"Processing {model_name}...")
        
        # Find metrics.csv file
        metrics_pattern = stats_dir / "model" / "log" / "csv" / f"{model_name}_supervised" / "version_*" / "metrics.csv"
        metrics_files = list(glob.glob(str(metrics_pattern)))
        
        if not metrics_files:
            print(f"  No metrics.csv found for {model_name}")
            continue
        
        # Get the latest version (highest version number)
        latest_metrics = max(metrics_files, key=lambda x: int(Path(x).parent.name.split('_')[-1]))
        print(f"  Using metrics file: {latest_metrics}")
        
        try:
            # Read metrics
            df = pd.read_csv(latest_metrics)
            
            # Extract train and val losses (handle empty cells)
            train_losses = df[df['loss'].notna() & (df['loss'] != '')]['loss'].copy()
            val_losses = df[df['val_loss'].notna() & (df['val_loss'] != '')]['val_loss'].copy()
            
            # Extract test loss if available
            test_loss_value = None
            if 'test_loss' in df.columns:
                test_losses = df[df['test_loss'].notna() & (df['test_loss'] != '')]['test_loss'].copy()
                if not test_losses.empty:
                    test_loss_value = pd.to_numeric(test_losses.iloc[0], errors='coerce')
            
            # Extract test NRMSE if available
            test_nrmse_value = None
            if 'test_nrmse' in df.columns:
                test_nrmse = df[df['test_nrmse'].notna() & (df['test_nrmse'] != '')]['test_nrmse'].copy()
                if not test_nrmse.empty:
                    test_nrmse_value = pd.to_numeric(test_nrmse.iloc[0], errors='coerce')
            
            # Convert to numeric, coercing errors
            train_losses = pd.to_numeric(train_losses, errors='coerce')
            val_losses = pd.to_numeric(val_losses, errors='coerce')
            
            # Drop NaN values
            train_losses = train_losses.dropna()
            val_losses = val_losses.dropna()
            
            if len(train_losses) == 0 and len(val_losses) == 0:
                print(f"  No valid loss data found for {model_name}")
                continue
            
            # Calculate min and std
            train_min = train_losses.min() if len(train_losses) > 0 else np.nan
            val_min = val_losses.min() if len(val_losses) > 0 else np.nan
            train_std = train_losses.std() if len(train_losses) > 1 else 0.0
            val_std = val_losses.std() if len(val_losses) > 1 else 0.0
            
            model_names.append(model_name)
            train_loss_min.append(train_min)
            val_loss_min.append(val_min)
            train_loss_std.append(train_std)
            val_loss_std.append(val_std)
            
            # Store test values if available
            if test_loss_value is not None and not pd.isna(test_loss_value):
                test_loss_values.append(test_loss_value)
            else:
                test_loss_values.append(None)
            
            if test_nrmse_value is not None and not pd.isna(test_nrmse_value):
                test_nrmse_values.append(test_nrmse_value)
            else:
                test_nrmse_values.append(None)
            
            print(f"  Train: min={train_min:.4f}, std={train_std:.4f}")
            print(f"  Val: min={val_min:.4f}, std={val_std:.4f}")
            if test_loss_value is not None and not pd.isna(test_loss_value):
                print(f"  Test Loss: {test_loss_value:.4f}")
            if test_nrmse_value is not None and not pd.isna(test_nrmse_value):
                print(f"  Test NRMSE: {test_nrmse_value:.4f}")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            continue
    
    if not model_names:
        print("No valid model data found!")
        return
    
    # Check if we have test NRMSE data for second plot
    has_test_nrmse = any(x is not None for x in test_nrmse_values)
    
    if has_test_nrmse:
        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    else:
        # Create single plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(model_names))
    
    # Find top 3 models for val_loss (lowest values)
    val_loss_with_indices = [(val_loss_min[i], i) for i in range(len(val_loss_min)) if val_loss_min[i] is not None]
    val_loss_with_indices.sort()  # Sort by val_loss (ascending)
    top3_val_indices = [idx for _, idx in val_loss_with_indices[:3]]
    
    print(f"Top 3 models by validation loss: {[model_names[i] for i in top3_val_indices]}")
    
    # Plot 1: Loss comparison
    ax1.errorbar(x_pos, train_loss_min, yerr=train_loss_std, 
                fmt='o-', label='Train Loss (min)', color='blue', capsize=5)
    ax1.errorbar(x_pos, val_loss_min, yerr=val_loss_std, 
                fmt='s-', label='Val Loss (min)', color='red', capsize=5)
    
    # Overlay top 3 val_loss markers in gold
    top3_x = [x_pos[i] for i in top3_val_indices]
    top3_val = [val_loss_min[i] for i in top3_val_indices]
    top3_val_std = [val_loss_std[i] for i in top3_val_indices]
    ax1.errorbar(top3_x, top3_val, yerr=top3_val_std, 
                fmt='s-', color='gold', capsize=5, markersize=8, 
                label='Top 3 Val Loss', zorder=5)
    
    # Add test loss if available
    test_loss_available = [x for x in test_loss_values if x is not None]
    if test_loss_available:
        test_loss_positions = [i for i, x in enumerate(test_loss_values) if x is not None]
        test_loss_vals = [x for x in test_loss_values if x is not None]
        ax1.plot(test_loss_positions, test_loss_vals, 'g--', label='Test Loss', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Model Names')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Comparison: Minimum Loss Across Epochs')
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test NRMSE comparison (if available)
    if has_test_nrmse:
        # Find top 3 models for test NRMSE (lowest values)
        test_nrmse_with_indices = [(test_nrmse_values[i], i) for i in range(len(test_nrmse_values)) if test_nrmse_values[i] is not None]
        test_nrmse_with_indices.sort()  # Sort by test NRMSE (ascending)
        top3_nrmse_indices = [idx for _, idx in test_nrmse_with_indices[:3]]
        
        print(f"Top 3 models by test NRMSE: {[model_names[i] for i in top3_nrmse_indices]}")
        
        test_nrmse_available = [x for x in test_nrmse_values if x is not None]
        if test_nrmse_available:
            test_nrmse_positions = [i for i, x in enumerate(test_nrmse_values) if x is not None]
            test_nrmse_vals = [x for x in test_nrmse_values if x is not None]
            ax2.plot(test_nrmse_positions, test_nrmse_vals, 'o-', color='green', 
                    linewidth=2, markersize=6, label='Test NRMSE')
            
            # Overlay top 3 test NRMSE markers in gold
            top3_nrmse_x = [x_pos[i] for i in top3_nrmse_indices]
            top3_nrmse_vals = [test_nrmse_values[i] for i in top3_nrmse_indices]
            ax2.plot(top3_nrmse_x, top3_nrmse_vals, 'o-', color='gold', 
                    linewidth=2, markersize=8, label='Top 3 Test NRMSE', zorder=5)
        
        ax2.set_xlabel('Model Names')
        ax2.set_ylabel('NRMSE')
        ax2.set_title('Model Comparison: Test NRMSE Across Models')
        ax2.set_ylim(0.95, 1.05)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = base_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {output_path}")

def plot_mae_reconstructions(original, reconstructed, mask, epoch, plots_dir, sample_idx=0):
    """
    Plot MAE reconstructions for visualization.
    
    Args:
        original: Original signal tensor (B, C, T)
        reconstructed: Reconstructed signal tensor (B, C, T) 
        mask: Boolean mask tensor (B, C, T)
        epoch: Current epoch number
        plots_dir: Directory to save plots
        sample_idx: Which sample to plot (default: 0)
    """
    # Convert to numpy
    if torch.is_tensor(original):
        original_np = original.cpu().numpy()
    else:
        original_np = original
        
    if torch.is_tensor(reconstructed):
        reconstructed_np = reconstructed.cpu().numpy()
    else:
        reconstructed_np = reconstructed
        
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Debug: Print data ranges
    print(f"Plot - Original range: [{original_np.min():.4f}, {original_np.max():.4f}]")
    print(f"Plot - Reconstructed range: [{reconstructed_np.min():.4f}, {reconstructed_np.max():.4f}]")
    print(f"Plot - Original mean: {original_np.mean():.4f}")
    print(f"Plot - Reconstructed mean: {reconstructed_np.mean():.4f}")
    
    # Get dimensions
    B, C, T = original_np.shape
    time_axis = np.arange(T)
    
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Individual channels for selected sample
    channels_to_plot = [0, C//2, C-1]  # First, middle, last channel
    
    for i, ch in enumerate(channels_to_plot):
        ax = axes[0, i]
        
        # Plot original
        ax.plot(time_axis, original_np[sample_idx, ch, :], 'b-', label='Original', alpha=0.7, linewidth=1)
        
        # Plot reconstruction
        ax.plot(time_axis, reconstructed_np[sample_idx, ch, :], 'r-', label='Reconstructed', alpha=0.7, linewidth=1)
        
        # Shade masked regions
        masked_regions = mask_np[sample_idx, ch, :]
        if masked_regions.any():
            # Simple approach: shade all masked regions
            for i, is_masked in enumerate(masked_regions):
                if is_masked:
                    ax.axvspan(i, i+1, alpha=0.3, color='gray', label='Masked' if i == 0 else "")
        
        ax.set_title(f'Channel {ch} | sample {sample_idx}')
        ax.set_xlabel(f'Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean across channels for first 3 samples
    for idx in range(min(3, B)):
        ax = axes[1, idx]
        
        # Compute mean across channels
        original_mean = np.mean(original_np[idx], axis=0)
        reconstructed_mean = np.mean(reconstructed_np[idx], axis=0)
        mask_mean = np.mean(mask_np[idx], axis=0) > 0.5  # Majority vote
        
        # Plot original
        ax.plot(time_axis, original_mean, 'b-', label='Original', alpha=0.7, linewidth=1)
        
        # Plot reconstruction
        ax.plot(time_axis, reconstructed_mean, 'r-', label='Reconstructed', alpha=0.7, linewidth=1)
        
        # Shade masked regions
        if mask_mean.any():
            # Simple approach: shade all masked regions
            for i, is_masked in enumerate(mask_mean):
                if is_masked:
                    ax.axvspan(i, i+1, alpha=0.3, color='gray', label='Masked' if i == 0 else "")
        
        ax.set_title(f'Sample {idx} (Mean across channels)')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Set consistent y-limits for better comparison
    all_values = np.concatenate([original_np.flatten(), reconstructed_np.flatten()])
    
    # Set consistent y-limits for better comparison
    y_min, y_max = np.percentile(all_values, [5, 95])
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)
    
    plt.suptitle(f'MAE Reconstruction - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = plots_dir / f'mae_reconstruction_epoch_{epoch}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MAE reconstruction plot saved to {plot_path}")

def plot_mae_reconstruction(original, reconstructed, mask, epoch, plots_dir, sample_idx=0):
    """
    Plot MAE reconstruction showing average signals across subjects/windows for different batches.
    
    Args:
        original: Original EEG signal [B, C, T]
        reconstructed: Reconstructed EEG signal [B, C, T] 
        mask: Mask indicating masked regions [B, C, T]
        epoch: Current epoch number
        plots_dir: Directory to save plots
        sample_idx: Index of sample to plot (default: 0)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Convert to numpy if needed
    if torch.is_tensor(original):
        original_np = original.cpu().numpy()
    else:
        original_np = original
        
    if torch.is_tensor(reconstructed):
        reconstructed_np = reconstructed.cpu().numpy()
    else:
        reconstructed_np = reconstructed
        
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Get dimensions
    B, C, T = original_np.shape
    time_axis = np.arange(T)
    
    # Create figure with 2 rows: channel averages and mean across all channels
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select representative channels
    channels_to_plot = [0, C//2, C-1]
    
    # First row: Average signal across subjects/windows for selected channels
    for i, ch in enumerate(channels_to_plot):
        ax = axes[0, i]
        
        # Average across batch dimension (subjects/windows)
        orig_avg = np.mean(original_np[:, ch, :], axis=0)  # [T]
        recon_avg = np.mean(reconstructed_np[:, ch, :], axis=0)  # [T]
        mask_avg = np.mean(mask_np[:, ch, :], axis=0) > 0.5  # [T]
        
        # Plot average signals
        ax.plot(time_axis, orig_avg, 'b-', label='Original (avg)', alpha=0.8, linewidth=1.5)
        ax.plot(time_axis, recon_avg, 'r-', label='Reconstructed (avg)', alpha=0.8, linewidth=1.5)
        
        # Shade masked regions
        if mask_avg.any():
            for t_idx, is_masked in enumerate(mask_avg):
                if is_masked:
                    ax.axvspan(t_idx, t_idx+1, alpha=0.3, color='gray', 
                              label='Masked' if t_idx == 0 else "")
        
        ax.set_title(f'Channel {ch} - Average across {B} subjects')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Second row: Average signal across all channels for different batch samples
    batch_samples = [0, min(1, B-1), min(2, B-1)]  # Up to 3 different batches
    
    for i, batch_idx in enumerate(batch_samples):
        ax = axes[1, i]
        
        # Mean across all channels for this batch sample
        orig_mean = np.mean(original_np[batch_idx], axis=0)  # [T]
        recon_mean = np.mean(reconstructed_np[batch_idx], axis=0)  # [T]
        mask_mean = np.mean(mask_np[batch_idx], axis=0) > 0.5  # [T]
        
        # Plot mean signals
        ax.plot(time_axis, orig_mean, 'b-', label='Original (mean)', alpha=0.8, linewidth=1.5)
        ax.plot(time_axis, recon_mean, 'r-', label='Reconstructed (mean)', alpha=0.8, linewidth=1.5)
        
        # Shade masked regions
        if mask_mean.any():
            for t_idx, is_masked in enumerate(mask_mean):
                if is_masked:
                    ax.axvspan(t_idx, t_idx+1, alpha=0.3, color='gray', 
                              label='Masked' if t_idx == 0 else "")
        
        ax.set_title(f'Batch {batch_idx} - Mean across {C} channels')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Set consistent y-limits for better comparison
    all_values = np.concatenate([original_np.flatten(), reconstructed_np.flatten()])
    y_min, y_max = np.percentile(all_values, [5, 95])
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)
    
    plt.suptitle(f'MAE Reconstruction Analysis - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = plots_dir / f'mae_reconstruction_analysis_epoch_{epoch}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAE reconstruction analysis plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses')
    parser.add_argument('--metrics_dir', help='Directory containing metrics.csv')
    parser.add_argument('--base_dir', help='Base directory containing stats_* folders for model comparison')
    parser.add_argument('--mode', choices=['single', 'comparison'], default='single', 
                       help='Plot mode: single model or model comparison')
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.metrics_dir:
            print("Error: --metrics_dir required for single mode")
            return
        plot_losses(args.metrics_dir)
    elif args.mode == 'comparison':
        if not args.base_dir:
            print("Error: --base_dir required for comparison mode")
            return
        plot_model_comparison(args.base_dir)


if __name__ == '__main__':
    main()
