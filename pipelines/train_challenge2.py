#!/usr/bin/env python3
"""
Challenge 2: Supervised Learning
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from eegchallenge.models import get_model, GenericModel, MultiTaskModel
from eegchallenge.parser import parse_args, get_config_from_args, config_call
from eegchallenge.data import load_and_process_data 
from eegchallenge.io import create_directories, print_config

def main(config):

    print_config(config, '1')
    
    # -------------------------------
    # Loading dataset
    # -------------------------------

    dirs = config_call(create_directories, config)
    # Prepare augmentation parameters
    augmentation_params = {"noise_std": config.get("noise_std", 0.1)} if config.get("enable_augmentation", False) else None
    
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=2, 
                                             augmentation_params=augmentation_params) # Seed is fixed here # Issue with dtype of output, change to float32
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # -------------------------------
    # Loading model
    # -------------------------------
    
    print(f'\nModel: {config['model_name']}')
    base_model = get_model(config['model_name'], n_chans=129, n_outputs=1, n_times=config['window_length']*100, sfreq=100)
    loss = torch.nn.MSELoss() # ADAPT HERE TO INCLUDE REGULARIZATION
    model = GenericModel(base_model, loss = loss, optimizer = 'adamw', optimizer_params = {'lr':config['learning_rate'], 'weight_decay':config['weight_decay']}, dtype=torch.float32, use_se = True, use_cdrop = True)#, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_params = {'T_max':config['num_epochs']-1})

    # -------------------------------
    # Setting logs
    # -------------------------------

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath = dirs['model'], filename = f'{config['model_name']}_supervised_best_{config['iter']}', auto_insert_metric_name = False, save_on_train_epoch_end = True, save_top_k = 1)
    logger = [
        pl.loggers.TensorBoardLogger(save_dir = dirs['model'] / 'log' / 'tensorboard', name=f'{config['model_name']}_supervised_{config['iter']}'),
        pl.loggers.CSVLogger(save_dir = dirs['model'] / 'log' / 'csv', name=f'{config['model_name']}_supervised_{config['iter']}')
    ]
    gpu_stats = pl.callbacks.DeviceStatsMonitor()
    
    # -------------------------------
    # Training model
    # -------------------------------

    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=True)
    
    # ADAPT HERE TO INCLUDE PRETRAINING - can use MultiTaskModel
    trainer = pl.Trainer(default_root_dir = dirs['model'] / 'log', callbacks = [checkpoint, early_stopping, gpu_stats], logger = logger, deterministic = True, accelerator = 'auto', max_epochs=config['num_epochs'])
    metrics = trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    import mne
    mne.set_log_level('WARNING')
    print('Challenge 2', flush=True)
    args = parse_args()
    config = get_config_from_args(args)
    main(config)
