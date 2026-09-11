#!/usr/bin/env python3
"""
Data Augmentation Module for EEG Data
"""

import numpy as np
import torch
import warnings
from PyEMD import EMD, EEMD

def _to_numpy_and_dtype(X):
    """Return (array, original_dtype, is_torch) for flexible inputs."""
    original_dtype = getattr(X, "dtype", np.float32)
    is_torch = False
    try:
        import torch
        if isinstance(X, torch.Tensor):
            is_torch = True
            original_dtype = X.dtype
            X = X.cpu().numpy()
    except Exception:
        pass
    # if it's a numpy-like object with .numpy()
    if hasattr(X, "numpy") and not is_torch:
        X = X.numpy()
    return np.asarray(X), original_dtype, is_torch

def _maybe_return_torch(arr, original_dtype, was_torch):
    """Convert back to torch tensor if needed, preserving dtype where possible."""
    if was_torch:
        import torch
        # attempt to convert dtype sensibly
        try:
            return torch.tensor(arr, dtype=original_dtype)
        except Exception:
            return torch.tensor(arr)
    else:
        # return numpy with same dtype if possible
        try:
            return arr.astype(original_dtype)
        except Exception:
            return arr

def gaussian_noise(X, noise_std=0.1, **kwargs):
    """Add Gaussian noise to EEG signal."""
    # Store original dtype
    original_dtype = X.dtype if hasattr(X, 'dtype') else np.float32
    
    # Convert to numpy if needed
    if hasattr(X, 'numpy'):
        X = X.numpy()
    elif torch.is_tensor(X):
        X = X.numpy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, X.shape).astype(original_dtype)
    return (X + noise).astype(original_dtype)

def channel_dropout(X, dropout_prob=0.1, **kwargs):
    """Randomly set some channels to zero (channel dropout)."""
    # Store original dtype
    original_dtype = X.dtype if hasattr(X, 'dtype') else np.float32
    
    # Convert to numpy if needed
    if hasattr(X, 'numpy'):
        X = X.numpy()
    elif torch.is_tensor(X):
        X = X.numpy()
    
    # Ensure X is 2D: [C, T]
    if X.ndim == 1:
        X = X[None, :]  # [T] -> [1, T]
    elif X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])  # [B, C, T] -> [C, T]
    
    # Create mask for channels to keep
    mask = np.random.random(X.shape[0]) > dropout_prob
    X_aug = X.copy().astype(original_dtype)
    X_aug[~mask] = 0
    
    return X_aug

def emd_eeg(X, noise_std=0.1, eemd=False, eemd_trials=50, random_state=None, max_perturb_imfs=2, clip_factor=5.0):
    X, orig_dtype, was_torch = _to_numpy_and_dtype(X)
    rng = np.random.default_rng(random_state)
    single_trace = (X.ndim == 1)
    if single_trace:
        X = X[None, None, :]
    elif X.ndim == 2:
        X = X[:, None, :]
    n_trials, n_channels, T = X.shape
    augmented = np.empty_like(X, dtype=float)

    if eemd:
        eemd_obj = EEMD()
        try:
            eemd_obj.trials = int(eemd_trials)
        except Exception:
            pass
        decompose = lambda s: eemd_obj.eemd(s) if hasattr(eemd_obj, "eemd") else eemd_obj(s)
    else:
        emd_obj = EMD()
        decompose = lambda s: emd_obj.emd(s)

    for i in range(n_trials):
        for ch in range(n_channels):
            sig = X[i, ch, :].astype(float)
            try:
                imfs = decompose(sig)
            except Exception as e:
                warnings.warn(f"EMD failed for trial {i}, ch {ch}: {e}. Returning original signal.")
                augmented[i, ch, :] = sig
                continue

            if imfs is None or imfs.size == 0:
                augmented[i, ch, :] = sig
                continue

            imfs = np.atleast_2d(imfs)
            n_imfs = imfs.shape[0]
            k = min(max_perturb_imfs, n_imfs)  # only perturb first k (highest freq) by default
            imf_stds = np.std(imfs[:k, :], axis=1) + 1e-16
            noise = rng.normal(0.0, 1.0, size=(k, T)) * (noise_std * imf_stds[:, None])
            noisy_imfs = imfs.copy()
            noisy_imfs[:k, :] = noisy_imfs[:k, :] + noise
            rec = np.sum(noisy_imfs, axis=0)
            # optional clipping to avoid extreme spikes: clip to mean±clip_factor*std(original)
            orig_std = sig.std()
            augmented[i, ch, :] = np.clip(rec, sig.mean() - clip_factor*orig_std, sig.mean() + clip_factor*orig_std)

    # restore input shape semantics
    if single_trace:
        augmented = augmented[0, 0, :]
    elif X.ndim == 2:
        augmented = augmented[:, 0, :]
    
    # Ensure consistent output shape
    if augmented.ndim == 3 and augmented.shape[1] == 1:
        augmented = augmented[:, 0, :]  # [C, 1, T] -> [C, T]
    
    return _maybe_return_torch(augmented, orig_dtype, was_torch)


# Registry of available augmentation functions
AUGMENTATION_FUNCTIONS = {
    'gaussian_noise': gaussian_noise,
    'channel_dropout': channel_dropout,
    'emd_eeg': emd_eeg,
}

def get_available_augmentations():
    """Get list of available augmentation functions."""
    return list(AUGMENTATION_FUNCTIONS.keys())

def apply_augmentation(augmentation_name, X, **params):
    """Apply specified augmentation to EEG signal."""
    if augmentation_name not in AUGMENTATION_FUNCTIONS:
        available = get_available_augmentations()
        raise ValueError(f"Augmentation '{augmentation_name}' not found. Available: {available}")
    
    augmentation_func = AUGMENTATION_FUNCTIONS[augmentation_name]
    return augmentation_func(X, **params)


if __name__ == "__main__":
    print("Available augmentations:", get_available_augmentations())