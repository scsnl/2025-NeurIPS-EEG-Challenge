#!/usr/bin/env python3
"""
I/O Helpers Module
"""
from pathlib import Path

def create_directories(results_folder, save_metrics, save_plots, model_name, use_all_releases):
    """Create necessary directories for outputs."""
    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    
    dirs = {"results": results_path}
    
    if save_metrics:
        metrics_dir = results_path / f"stats_{model_name}" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        dirs["metrics"] = metrics_dir
    
    if save_plots:
        plots_dir = results_path / f"stats_{model_name}" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dirs["plots"] = plots_dir

    model_dir = results_path / f"stats_{model_name}" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dirs["model"] = model_dir
    
    return dirs

def save_metrics(metrics_df, metrics_dir, filename="training_metrics.csv"):
    """Save training metrics to CSV."""
    filepath = metrics_dir / filename
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")

def print_config(config, challenge):
    print("=" * 80)
    print(f"CHALLENGE {challenge}: CROSS-TASK TRANSFER LEARNING")
    print("=" * 80)
    print(f"Data folder: {config['data_folder']}")
    print(f"Results folder: {config['results_folder']}")
    print(f"Task name: {config['task_name']}")
    print(f"Objective: {config['objective']}")
    print(f"Model: {config['model_name']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Window length: {config['window_length']}s")
    print(f"Shift: {config['shift']}s")
    print(f"Max subjects: {config['max_subjects']}")
    print(f"Data releases: {config['data_releases']}")
    print(f"Use all releases: {config['use_all_releases']}")
    print(f"enable_augmentation: {config['enable_augmentation']}")
    print(f"augmentation_name: {config['augmentation_name']}")
    print(f"normalize_data: {config["normalize_data"]}")
    print(f"Enable CV: {config['enable_cv']}")
    if config['enable_cv']:
        print(f"CV folds: {config['cv_folds']}")
    print(f"Seed: {config['seed']}")
    print("=" * 80)

