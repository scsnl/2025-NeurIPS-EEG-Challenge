#!/usr/bin/env python3
"""
Challenge 1: MAE-style Pretraining
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from eegchallenge.models import get_model, MAEPretrainer
from eegchallenge.parser import parse_args, get_config_from_args, config_call
from eegchallenge.data import load_and_process_data, SimpleWrapper 
from eegchallenge.io import create_directories, print_config


def main(config):
    print_config(config, '1_mae')
    
    # -------------------------------
    # Loading dataset
    # -------------------------------
    
    dirs = config_call(create_directories, config)
    
    # Load data for pretraining (use all available data)
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=2)
    
    
    # Apply robust scaling and normalization to all datasets
    normalize = config.get('normalize_data', False)
    print(f"normalize: {normalize}")
    stats_n_samples = config.get('stats_n_samples', 200)
    stats_seed = config.get('stats_seed', 42)
    
    # Create wrappers with consistent scaling and per-channel normalization
    train_set = SimpleWrapper(train_set, scale_factor=1e3, normalize=normalize, 
                             stats_n_samples=stats_n_samples, stats_seed=stats_seed)
    
    # For val/test, use same normalization stats as training (no data leaks)
    if normalize:
        val_set = SimpleWrapper(val_set, scale_factor=1e3, normalize=False, 
                               stats_n_samples=stats_n_samples, stats_seed=stats_seed)
        val_set.global_mean = train_set.global_mean
        val_set.global_std = train_set.global_std
        val_set.normalize = True
        
        test_set = SimpleWrapper(test_set, scale_factor=1e3, normalize=False, 
                                stats_n_samples=stats_n_samples, stats_seed=stats_seed)
        test_set.global_mean = train_set.global_mean
        test_set.global_std = train_set.global_std
        test_set.normalize = True
    else:
        val_set = SimpleWrapper(val_set, scale_factor=1e3, normalize=False, 
                               stats_n_samples=stats_n_samples, stats_seed=stats_seed)
        test_set = SimpleWrapper(test_set, scale_factor=1e3, normalize=False, 
                               stats_n_samples=stats_n_samples, stats_seed=stats_seed)
    
    # Test data consistency
    print("Testing data consistency...")
    train_sample = train_set[0]
    val_sample = val_set[0]
    print(f"Train sample shape: {train_sample[0].shape if isinstance(train_sample, tuple) else train_sample.shape}")
    print(f"Val sample shape: {val_sample[0].shape if isinstance(val_sample, tuple) else val_sample.shape}")
    print(f"Train sample range: [{train_sample[0].min():.4f}, {train_sample[0].max():.4f}]" if isinstance(train_sample, tuple) else f"Train sample range: [{train_sample.min():.4f}, {train_sample.max():.4f}]")
    print(f"Val sample range: [{val_sample[0].min():.4f}, {val_sample[0].max():.4f}]" if isinstance(val_sample, tuple) else f"Val sample range: [{val_sample.min():.4f}, {val_sample.max():.4f}]")
    
    # Create optimized data loaders for faster pretraining
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'],pin_memory=True,persistent_workers=True if config['num_workers'] > 0 else False,drop_last=True,prefetch_factor=2 * config['num_workers'] if config['num_workers'] > 0 else 2)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],pin_memory=True,persistent_workers=True if config['num_workers'] > 0 else False,drop_last=True,prefetch_factor=2 * config['num_workers'] if config['num_workers'] > 0 else 2)
    
    # -------------------------------
    # Loading model
    # -------------------------------
    
    print(f'\nModel: {config['model_name']}')
    print(f'Decoder: {config.get('decoder', 'conv')}')
    print(f'Patch size: {config.get('patch_size', 50)}')
    print(f'Mask ratio: {config.get('mask_ratio', 0.5)}')
    print(f'Normalize data: {config.get('normalize_data', False)}')
    print(f'Learning rate: {config['learning_rate']}')
    print(f'Weight decay: {config['weight_decay']}')

    def create_mae_pretrainer(plots_dir: str, model_name: str, n_chans: int = 129, n_outputs: int = 1, 
                            n_times: int = 2000, sfreq: int = 100, patch_size: int = 50, 
                            mask_ratio: float = 0.5, mask_seed: int = None, decoder: str = 'conv',
                            learning_rate: float = 1e-4, weight_decay: float = 1e-5, 
                            num_epochs: int = 100):
        # Get base model and pretrainer
        backbone = get_model(model_name, n_chans=n_chans,n_outputs=n_outputs,n_times=n_times,sfreq=sfreq)
        pretrainer = MAEPretrainer(plots_dir, backbone, patch_size=patch_size,mask_ratio=mask_ratio,mask_seed=mask_seed,decoder=decoder,learning_rate=learning_rate,weight_decay=weight_decay,num_epochs=num_epochs)
        return pretrainer
    
    # Create MAE pretrainer
    plots_dir = dirs['model'] / 'log' / 'plots'
    model = create_mae_pretrainer(plots_dir=plots_dir, model_name=config['model_name'],n_chans=129,n_outputs=1,n_times=config['window_length']*100,sfreq=100,
        patch_size=config.get('patch_size', 50),mask_ratio=config.get('mask_ratio', 0.5),mask_seed=config.get('mask_seed', None),decoder=config.get('decoder', 'conv'),
        learning_rate=config['learning_rate'],weight_decay=config['weight_decay'],num_epochs=config['num_epochs'])
    
    print("Initializing decoder with dummy forward pass...")
    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        if isinstance(dummy_batch, (list, tuple)):
            dummy_x = dummy_batch[0]
        else:
            dummy_x = dummy_batch
        dummy_x = dummy_x[:1].to(model.device)  # Take only first sample
        _ = model.forward(dummy_x, training=False)
    print("Decoder initialization complete.")
    
    # -------------------------------
    # Setting logs
    # -------------------------------
    
    # Checkpoint callback
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirs['model'], 
                                            filename=f'{config['model_name']}_mae_pretraining_best', 
                                            auto_insert_metric_name=False, 
                                            save_on_train_epoch_end=True, 
                                            save_top_k=1)
    # Logger
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=dirs['model'] / 'log' / 'tensorboard', 
                                    name=f'{config['model_name']}_mae_pretraining'),
        pl.loggers.CSVLogger(save_dir=dirs['model'] / 'log' / 'csv', 
                            name=f'{config['model_name']}_mae_pretraining')
    ]
    
    # Early stopping
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',patience=config.get('early_stopping_patience', 20),min_delta=0.001,mode='min',verbose=True)
    gpu_stats = pl.callbacks.DeviceStatsMonitor()
    
    # -------------------------------
    # Training model
    # -------------------------------
    
    # Check for existing checkpoint
    resume_dir = dirs['model']
    latest_checkpoint = None
    if resume_dir.exists():
        checkpoints = list(resume_dir.glob('*.ckpt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Found checkpoint to resume from: {latest_checkpoint}")
    
    # Trainer
    trainer = pl.Trainer(default_root_dir=dirs['model'] / 'log', 
                        callbacks=[checkpoint, early_stopping, gpu_stats], 
                        logger=logger, 
                        accelerator='auto',
                        devices=2,
                        strategy='ddp_find_unused_parameters_true',
                        max_epochs=config['num_epochs'],
                        accumulate_grad_batches=4)
    
    # Train
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        try:
            print("Attempting to resume training from checkpoint...")
            trainer.fit(model, train_loader, val_loader, ckpt_path=str(latest_checkpoint))
        except RuntimeError as e:
            if "state_dict" in str(e) or "Unexpected key" in str(e):
                print(f"Checkpoint incompatible with current model: {e}")
                print("Starting fresh training instead...")
                trainer.fit(model, train_loader, val_loader)
            else:
                raise e
    else:
        print("Starting fresh training")
        trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = dirs['model'] / f'{config['model_name']}_mae_pretraining_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"Pretrained model saved to: {final_path}")
    print(f"\nPretraining completed!")


if __name__ == '__main__':
    import mne
    mne.set_log_level('WARNING')
    print('Challenge 1 MAE Pretraining', flush=True)
    args = parse_args()
    config = get_config_from_args(args)
    main(config)
