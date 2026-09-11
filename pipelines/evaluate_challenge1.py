#!/usr/bin/env python3
"""
Challenge 1: Supervised Learning
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob
import pandas as pd
from pathlib import Path

from eegchallenge.models import get_model, GenericModel, MultiTaskModel, NRMSELoss, OutputVariance
from eegchallenge.parser import parse_args, get_config_from_args, config_call
from eegchallenge.data import load_and_process_data 
from eegchallenge.io import create_directories, print_config

def find_model(path, version = ''):
    print(f'Trying to load model from {path}')
    l = glob.glob(f'{path}/*{version}.pt')
    if len(l)>0:
        return torch.load(l[-1]), 0
    l = glob.glob(f'{path}/*{version}.ckpt')
    if len(l)>0:
        return torch.load(l[-1])['state_dict'], 1
    raise NameError(f'Model (*.pt or *.ckpt) not found in {path}')

def main(config, args):

    print_config(config, '1')
    
    # -------------------------------
    # Loading dataset
    # -------------------------------

    dirs = config_call(create_directories, config)
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=1, existing_cache_key="c318a28d36d8c67f51d37a67b631928d") # Seed is fixed here # Issue with dtype of output, change to float32
    
    #train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    #val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # -------------------------------
    # Loading model
    # -------------------------------
    
    print(f'\nModel: {config['model_name']}')
    base_model = get_model(config['model_name'], n_chans=129, n_outputs=1, n_times=config['window_length']*100, sfreq=100)
    loss = torch.nn.MSELoss() # ADAPT HERE TO INCLUDE REGULARIZATION
    model = GenericModel(base_model, loss = loss, optimizer = 'adamw', optimizer_params = {'lr':config['learning_rate'], 'weight_decay':config['weight_decay']}, dtype=torch.float32, use_se = True, use_cdrop = True)#, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_params = {'T_max':config['num_epochs']-1})
    variance = OutputVariance()

    # -------------------------------
    # Loading model
    # -------------------------------
    
    state_dict, version = find_model(dirs['model'] if args.path is None else args.path, version = args.version)
    model(torch.zeros((1,129,200)))
    if version == 0:
        model.model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    gpu_stats = pl.callbacks.DeviceStatsMonitor() # log gpu stats

    # -------------------------------
    # Training model
    # -------------------------------
    
    trainer = pl.Trainer(default_root_dir = dirs['model'] / 'log', callbacks = [gpu_stats], deterministic = True, accelerator = 'auto')
    metrics, = trainer.test(model, test_loader)
    metrics.update(trainer.test(variance, test_loader)[0])
    print(f'NRMSE: {np.sqrt(metrics['test_loss']/metrics['output_variance']):.5f}')
    nrmse_final = np.sqrt(metrics['test_loss']/metrics['output_variance'])
    try:
        torch.save(model.model.state_dict(), f"{dirs['model']}/weights_challenge_1_{config['model_name']}.pt")
        print('Saved .pt')
    except:
        torch.save(state_dict, f"{dirs['model']}/weights_challenge_1_{config['model_name']}.pt")
        print('Saved state_dict')
    # Log to metrics.csv
    metrics_pattern = dirs['model'] / 'log' / 'csv' / f'{config['model_name']}_supervised' / 'version_*' / 'metrics.csv'
    metrics_files = list(glob.glob(str(metrics_pattern)))    
    if not metrics_files:
        print(f"  No metrics.csv found for {config['model_name']}")
    # Get the latest version (highest version number)
    metrics_file = max(metrics_files, key=lambda x: int(Path(x).parent.name.split('_')[-1]))
    df = pd.read_csv(metrics_file)
    df["test_loss"] = metrics['test_loss']  # Or use pd.NA for explicit missing data
    df["test_nrmse"] = nrmse_final
    df.to_csv(metrics_file, index=False)

    print(f"Updated CSV saved to {metrics_file}")

if __name__ == '__main__':
    import mne
    mne.set_log_level('WARNING')
    print('Challenge 1', flush=True)
    def add_argument(parser):
        parser.add_argument('--version', type=str, default='')
        parser.add_argument('--path', type=str, default= None)
        return parser
    args = parse_args(extra = add_argument)
    config = get_config_from_args(args)
    main(config, args)
