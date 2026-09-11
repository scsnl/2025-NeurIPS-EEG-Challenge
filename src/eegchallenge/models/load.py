#!/usr/bin/env python3
"""
Braindecode Models Module
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import warnings
from collections import OrderedDict

# Braindecode model imports
from braindecode.models import (
    ATCNet, AttnSleep, BIOT, CTNet, Deep4Net, EEGConformer, EEGNet, EEGNeX,
    EEGSimpleConv, FBCNet, Labram, MSVTNet, SignalJEPA, SPARCNet, SyncNet, TSception
)

# EEGPT model imports
from .eegpt import EEGPTWrapper
from .biot_pre import BIOT_PRE
from .cnn import eConvNet, AsdNet, stConvNet, ConvNet

MODEL_REGISTRY = {
    "ATCNet": ATCNet,
    "AttnSleep": AttnSleep,
    "BIOT": BIOT,
    "CTNet": CTNet,
    "Deep4Net": Deep4Net,
    "EEGConformer": EEGConformer,
    "EEGNet": EEGNet,
    "EEGNeX": EEGNeX,
    "EEGSimpleConv": EEGSimpleConv,
    "FBCNet": FBCNet,
    "Labram": Labram,
    "MSVTNet": MSVTNet,
    "SignalJEPA": SignalJEPA,
    "SPARCNet": SPARCNet,
    "SyncNet": SyncNet,
    "TSception": TSception,
    "linear": None,
    "EEGPT": EEGPTWrapper,
    "BIOT_PRE": BIOT_PRE,
    "eConvNet": eConvNet,
    "AsdNet": AsdNet,
    "stConvNet": stConvNet,
    "ConvNet": ConvNet,
}

def get_available_models():
    return list(MODEL_REGISTRY.keys())

def create_channel_info(n_chans, sfreq):
    """Create basic channel info for models that need it."""
    return {
        "ch_names": [f"EEG{i:03d}" for i in range(n_chans)],
        "sfreq": sfreq,
        "n_channels": n_chans
    }

def get_model(model_name, n_chans, n_outputs, n_times, sfreq=100):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {get_available_models()}")
    
    model_class = MODEL_REGISTRY[model_name]

    n_chans, n_outputs, n_times = int(n_chans), int(n_outputs), int(n_times)
    
    # Handle special cases
    if model_name == "SignalJEPA":
        chs_info = create_channel_info(n_chans, sfreq)
        model = model_class(n_times=n_times, chs_info=chs_info)
    elif model_name == ["BIOT", "AttnSleep", "FBCNet", "TSception", "EEGNeX", "BIOT_PRE"]:
        model = model_class(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    elif model_name == "EEGSimpleConv":
        model = model_class(n_chans=n_chans, n_outputs=n_outputs, sfreq=sfreq)
    elif model_name == "EEGPT":
        # LJ testing... After testing we can combine EEGPT with other functions above too
        # raise NotImplementedError(f"EEGPT model not implemeted yet")
        model = model_class(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    elif model_name == "linear":
        model = nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('output', torch.nn.Linear(n_chans*n_times, n_outputs)),
        ]))  
    elif model_name == ["eConvNet", "AsdNet", "stConvNet", "ConvNet"]:
        model = model_class()
    else:
        # Standard models
        model = model_class(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    
    print(f"Created {model_name} model")
    return model

if __name__ == "__main__":
    print("Available models:", get_available_models())
    try:
        model = get_model("EEGNeX", n_chans=129, n_outputs=1, n_times=200, sfreq=100)
        print(f"EEGNeX created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"Error: {e}")
