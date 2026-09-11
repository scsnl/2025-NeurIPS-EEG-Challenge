#!/usr/bin/env python3
"""
Challenge 1: Supervised Learning
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob

from eegchallenge.models import get_model, GenericModel, MultiTaskModel, NRMSELoss, OutputVariance
from eegchallenge.parser import parse_args, get_config_from_args, config_call
from eegchallenge.data import load_and_process_data 
from eegchallenge.io import create_directories, print_config

def find_model(path, version = ''):
    print(f'Trying to load model from {path}')
    #l = glob.glob(f'{path}/*{version}.pt')
    #if len(l)>0:
    #    return torch.load(l[-1]), 0
    l = glob.glob(f'{path}/*{version}.ckpt')
    if len(l)>0:
        return torch.load(l[-1])['state_dict'], 1
    raise NameError(f'Model (*.pt or *.ckpt) not found in {path}')

def main(config, args):

    print_config(config, '2')
    
    # -------------------------------
    # Loading dataset
    # -------------------------------

    dirs = config_call(create_directories, config)
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=2) # Seed is fixed here # Issue with dtype of output, change to float32
    
    #train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    #val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # -------------------------------
    # Loading model
    # -------------------------------
    
    print(f'\nModel: {config['model_name']}')
    base_model = get_model(config['model_name'], n_chans=129, n_outputs=1, n_times=config['window_length']*100, sfreq=100)
    loss = torch.nn.MSELoss() # ADAPT HERE TO INCLUDE REGULARIZATION
    model = GenericModel(base_model, loss = loss, metrics = {'nrmse': NRMSELoss()}, show = ['nrmse'], optimizer = 'adamw', optimizer_params = {'lr':config['learning_rate'], 'weight_decay':config['weight_decay']})#, scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_params = {'T_max':config['num_epochs']-1})
    variance = OutputVariance()

    # -------------------------------
    # Loading model
    # -------------------------------
    
    state_dict, version = find_model(dirs['model'] if args.path is None else args.path, version = args.version)
    if version == 0:
        model.model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # -------------------------------
    # Training model
    # -------------------------------
    
    trainer = pl.Trainer(default_root_dir = dirs['model'] / 'log', deterministic = True, accelerator = 'auto')
    metrics, = trainer.test(model, test_loader)
    metrics.update(trainer.test(variance, test_loader)[0])
    print(f'NRMSE: {np.sqrt(metrics['test_loss']/metrics['output_variance']):.5f}')
    try:
        torch.save(model.model.state_dict(), f"{dirs['model']}/weights_challenge_2_{config['model_name']}.pt")
        print('Saved .pt')
    except:
        torch.save(state_dict, f"{dirs['model']}/weights_challenge_2_{config['model_name']}.pt")

if __name__ == '__main__':
    import mne
    mne.set_log_level('WARNING')
    print('Challenge 2', flush=True)
    def add_argument(parser):
        parser.add_argument('--version', type=str, default='')
        parser.add_argument('--path', type=str, default= None)
        return parser
    args = parse_args(extra = add_argument)
    config = get_config_from_args(args)
    main(config, args)
