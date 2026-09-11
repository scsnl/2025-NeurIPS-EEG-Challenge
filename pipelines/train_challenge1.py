#!/usr/bin/env python3
"""
Challenge 1: Supervised Learning
"""
# GPU stats monitor added

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from eegchallenge.models import get_model, GenericModel, MultiTaskModel
from eegchallenge.parser import parse_args, get_config_from_args, config_call
from eegchallenge.data import load_and_process_data, SimpleWrapper
from eegchallenge.io import create_directories, print_config

def main(config):

    print_config(config, '1')
    
    # -------------------------------
    # Loading dataset
    # -------------------------------

    dirs = config_call(create_directories, config)
    # Prepare augmentation parameters
    augmentation_params = {"noise_std": config.get("noise_std", 0.1)} if config.get("enable_augmentation", False) else None
    
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=1, 
                                             augmentation_params=augmentation_params) # Seed is fixed here # Issue with dtype of output, change to float32

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
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'],pin_memory=True,persistent_workers=True if config['num_workers']>0 else False,drop_last=True,prefetch_factor=2*config['num_workers'] if config['num_workers']>0 else 2)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],pin_memory=True,persistent_workers=True if config['num_workers']>0 else False,drop_last=True,prefetch_factor=2*config['num_workers'] if config['num_workers']>0 else 2)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],pin_memory=True,persistent_workers=True if config['num_workers']>0 else False,drop_last=True,prefetch_factor=2*config['num_workers'] if config['num_workers']>0 else 2)
    
    # -------------------------------
    # Loading model
    # -------------------------------
    
    print(f'\nModel: {config['model_name']}', flush=True)
    base_model = get_model(config['model_name'], n_chans=129, n_outputs=1, n_times=config['window_length']*100, sfreq=100)
    loss = torch.nn.MSELoss() # ADAPT HERE TO INCLUDE REGULARIZATION
    model = GenericModel(base_model, loss = loss, optimizer = 'adamw', optimizer_params = {'lr':config['learning_rate'], 'weight_decay':config['weight_decay']}, dtype=torch.float32, use_se = True, use_cdrop = True)#, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_params = {'T_max':config['num_epochs']-1})

    # -------------------------------
    # Setting logs
    # -------------------------------

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath = dirs['model'], filename = f'{config['model_name']}_supervised_best', auto_insert_metric_name = False, save_on_train_epoch_end = True, save_top_k = 1)
    logger = [
        pl.loggers.TensorBoardLogger(save_dir = dirs['model'] / 'log' / 'tensorboard', name=f'{config['model_name']}_supervised'),
        pl.loggers.CSVLogger(save_dir = dirs['model'] / 'log' / 'csv', name=f'{config['model_name']}_supervised')
    ]
    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',patience=20,min_delta=0.001,mode='min',verbose=True)
    gpu_stats = pl.callbacks.DeviceStatsMonitor()
    
    # -------------------------------
    # Training model
    # -------------------------------
    
    # ADAPT HERE TO INCLUDE PRETRAINING - can use MultiTaskModel
    trainer = pl.Trainer(default_root_dir = dirs['model'] / 'log', callbacks = [checkpoint, early_stopping, gpu_stats], logger = logger, deterministic = True, accelerator = 'auto', max_epochs=config['num_epochs'])
    metrics = trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    import mne
    mne.set_log_level('WARNING')
    print('Challenge 1', flush=True)
    args = parse_args()
    config = get_config_from_args(args)
    main(config)
