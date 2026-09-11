#!/usr/bin/env python3
"""
Argument Parser Helper Module
"""

import argparse
import inspect
from .models import get_available_models

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_folder", type=str, default="local_R1", help="Data folder path")
    parser.add_argument("--results_folder", type=str, default="results", help="Results folder path")
    parser.add_argument("--task_name", type=str, default="contrastChangeDetection", help="Task name")
    parser.add_argument("--objective", type=str, default=None, help="Objective to predict (age, gender, pf, rt, perf)")
    parser.add_argument("--max_subjects", type=int, default=None, help="Max subjects to use")
    parser.add_argument("--use_all_releases", action="store_true", help="Use all data releases")
    parser.add_argument("--data_releases", type=str, nargs="+", default=None, help="Specific releases")
    parser.add_argument("--eval_release", type=str, default="R5", help="Evaluation release")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="EEGNeX", choices=get_available_models(), help="Model name")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--window_length", type=float, default=2.0, help="Window length (s)")
    parser.add_argument("--shift", type=float, default=0.5, help="Shift after stimulus (s)")
    parser.add_argument("--crop_size", type=float, default=2.0)
    
    # Training parameters
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Min delta for early stopping")
    parser.add_argument("--valid_frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--seed", type=int, default=1234)
    
    # Cross-validation
    parser.add_argument("--enable_cv", action="store_true", help="Enable cross-validation")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    
    # Augmentation
    parser.add_argument("--enable_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--augmentation_name", type=str, default="gaussian_noise", help="Augmentation method name")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise std for augmentation")
    
    # Normalization
    parser.add_argument("--normalize_data", action="store_true", help="Enable z-score normalization")
    
    # Output
    parser.add_argument("--save_metrics", action="store_true", default=True, help="Save metrics")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--save_model", action="store_true", default=True, help="Save model")
    
    # Dataset caching
    parser.add_argument("--cache_dataset", action="store_true", help="Save preprocessed dataset to cache")
    parser.add_argument("--load_cached_dataset", action="store_true", help="Load preprocessed dataset from cache")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache", help="Directory for dataset cache")
    
    # Multi-task pretraining
    parser.add_argument("--task_list", type=str, nargs='+', help="List of tasks for pretraining")
    
    # MAE pretraining
    parser.add_argument("--decoder", type=str, default="transformer", choices=["conv", "transformer"], help="Decoder type for MAE")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch size for masking")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask ratio for MAE")
    parser.add_argument("--mask_seed", type=int, default=1234, help="Seed for masking (None for random)")
    
    # System
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/auto)")
    parser.add_argument("--iter", type=str, default=1)
    
    return parser

def parse_args(extra = lambda x:x):
    parser = extra(create_parser())
    args = parser.parse_args()
    
    # Set default data releases
    if args.use_all_releases:
        args.data_releases = ["R1", "R2", "R3", "R4", 
                             "R6", "R7", "R8", 
                             "R9", "R10", "R11"]
    elif args.data_releases is None:
        args.data_releases = [args.eval_release]
    
    # Ensure task_list is always a list
    if args.task_list is not None:
        args.task_list = [args.task_list] if isinstance(args.task_list, str) else args.task_list

    print(args.task_list)
    
    return args

def get_config_from_args(args):
    """Convert args to config dictionary."""
    return {
        "data_folder": args.data_folder,
        "results_folder": args.results_folder,
        "task_name": args.task_name,
        "objective": args.objective,
        "max_subjects": args.max_subjects,
        "data_releases": args.data_releases,
        "eval_release": args.eval_release,
        "use_all_releases": args.use_all_releases,
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "window_length": args.window_length,
        "shift": getattr(args, 'shift', 0.5),
        "crop_size": getattr(args, 'crop_size', 2.0),
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "valid_frac": args.valid_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "enable_cv": args.enable_cv,
        "cv_folds": args.cv_folds,
        "enable_augmentation": args.enable_augmentation,
        "augmentation_name": args.augmentation_name,
        "noise_std": args.noise_std,
        "normalize_data": args.normalize_data,
        "save_metrics": args.save_metrics,
        "save_plots": args.save_plots,
        "save_model": args.save_model,
        "cache_dataset": args.cache_dataset,
        "load_cached_dataset": args.load_cached_dataset,
        "cache_dir": args.cache_dir,
        "num_workers": args.num_workers,
        "device": args.device,
        "iter": args.iter,
        "task_list": args.task_list,
        "decoder": args.decoder,
        "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio,
        "mask_seed": args.mask_seed,
    }

def config_call(f, config, **kwargs):
    return f(**kwargs, **{k:config[k] for k in inspect.signature(f).parameters if k in config})

if __name__ == "__main__":
    print("Available models:", get_available_models())
