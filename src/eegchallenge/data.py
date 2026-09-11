#!/usr/bin/env python3
"""
Data Loading Module
"""

import random
import pickle
import hashlib
from pathlib import Path
#import warnings
#warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import check_random_state

# Braindecode imports
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events, create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseDataset

# EEGDash imports for local data loading
try:
    from eegdash.dataset import EEGChallengeDataset
    from eegdash.hbn.windows import (
        annotate_trials_with_target,
        add_aux_anchors,
        add_extras_columns,
        keep_only_recordings_with,
    )
    EEGDASH_AVAILABLE = True
except ImportError:
    print("Warning: EEGDash not available. Using local BIDS data loading.")
    EEGDASH_AVAILABLE = False

# Default problematic subjects to remove (same as original scripts)
PROBLEMATIC_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"
]

def generate_cache_key(data_folder, task_name, data_releases, eval_release, use_all_releases, 
                      challenge, max_subjects, window_length, shift, crop_size, sfreq, 
                      valid_frac, test_frac, enable_cv, cv_folds, seed, enable_augmentation=False,
                      augmentation_name="gaussian_noise", augmentation_params=None, task_list=None):
    """Generate a unique cache key based on data loading parameters."""
    # Handle task_list for multitask scenarios
    if task_list is not None:
        task_name = f"multitask_{sorted(task_list)}"
    
    key_string = f"{data_folder}_{task_name}_{data_releases}_{eval_release}_{use_all_releases}_{challenge}_{max_subjects}_{window_length}_{shift}_{crop_size}_{sfreq}_{valid_frac}_{test_frac}_{enable_cv}_{cv_folds}_{seed}"
    
    # Only include augmentation parameters if augmentation is enabled
    if enable_augmentation:
        aug_params_str = str(sorted(augmentation_params.items())) if augmentation_params else "None"
        key_string += f"_{enable_augmentation}_{augmentation_name}_{aug_params_str}"
    
    return hashlib.md5(key_string.encode()).hexdigest()

def save_dataset_cache(datasets, cache_key, cache_dir):
    """Save preprocessed datasets to cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_path / f"dataset_{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(datasets, f)
        print(f"Dataset cache saved to {cache_file}")
        return True
    except Exception as e:
        print(f"Failed to save dataset cache: {e}")
        return False

def load_dataset_cache(cache_key, cache_dir):
    """Load preprocessed datasets from cache."""
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"dataset_{cache_key}.pkl"
    
    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            datasets = pickle.load(f)
        print(f"Dataset cache loaded from {cache_file}")
        return datasets
    except Exception as e:
        print(f"Failed to load dataset cache: {e}")
        return None

def load_single_release_data(data_folder, task_name, release, max_subjects=None):
    """Load data from a single release."""
    data_path = Path(data_folder)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    
    if EEGDASH_AVAILABLE:
        dataset = EEGChallengeDataset(
            task=task_name,
            release=release,
            cache_dir=data_path,
            mini=False
        )
        return dataset
    else:
        raise NotImplementedError("Custom BIDS loading not yet implemented. Please install EEGDash.")

def load_multiple_releases_data(data_folder, task_name, releases, max_subjects=None):
    """Load data from multiple releases."""
    all_datasets = []
    
    for release in releases:
        try:
            print(f"Loading release: {release}")
            dataset = load_single_release_data(data_folder, task_name, release, max_subjects)
            all_datasets.append(dataset)
            print(f"Successfully loaded {len(dataset)} recordings from {release}")
        except Exception as e:
            print(f"Warning: Failed to load release {release}: {str(e)}")
            continue
    
    if not all_datasets:
        raise RuntimeError("No datasets were successfully loaded")
    
    combined_dataset = BaseConcatDataset(all_datasets)
    print(f"Combined dataset contains {len(combined_dataset)} recordings from {len(releases)} releases")
    
    return combined_dataset

def process_challenge_1_data(dataset, window_length=2.0, shift=0.5, sfreq=100, seed=None):
    """Process data for Challenge 1 (Cross-Task Transfer Learning)."""
    # Data preprocessing (same as original script)
    EPOCH_LEN_S = window_length
    
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
            require_stimulus=True, require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset, transformation_offline, n_jobs=1)
    
    ANCHOR = "stimulus_anchor"
    SHIFT_AFTER_STIM = shift
    WINDOW_LEN = window_length
    
    # Keep only recordings that actually contain stimulus anchors
    dataset = keep_only_recordings_with(ANCHOR, dataset)
    
    # Create single-interval windows
    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * sfreq),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * sfreq),
        window_size_samples=int(EPOCH_LEN_S * sfreq),
        window_stride_samples=sfreq,
        preload=False, # NB changed 2025-10-24
    )
    
    # Inject metadata
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )
    
    return single_windows

class DatasetWrapper(BaseDataset):
    """Wrapper class for EEG windows dataset with p-factor labels."""
    
    def __init__(self, dataset, crop_size_samples, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y, crop_inds = self.dataset[index]

        # P-factor label:
        #p_factor = self.dataset.description["p-factor"]
        p_factor = self.dataset.description["externalizing"]
        p_factor = float(p_factor)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        if self.crop_size_samples is not None:
            i_window_in_trial, i_start, i_stop = crop_inds
            assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
            start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
            i_start = i_start + start_offset
            i_stop = i_start + self.crop_size_samples
            X = X[:, start_offset : start_offset + self.crop_size_samples]
            crop_inds = (i_window_in_trial, i_start, i_stop)

        return X, p_factor

class AugmentationWrapper(BaseDataset):
    """Wrapper to add augmentation to any dataset while maintaining the same interface."""
    
    def __init__(self, dataset, enable_augmentation=False, augmentation_name="gaussian_noise", 
                 augmentation_params=None):
        self.dataset = dataset
        self.enable_augmentation = enable_augmentation
        self.augmentation_name = augmentation_name
        self.augmentation_params = augmentation_params or {}
        
        # Import augmentation function if needed
        if self.enable_augmentation:
            from .models.augmentation import apply_augmentation
            self.apply_augmentation = apply_augmentation

    def __len__(self):
        # If augmentation is enabled, double the dataset size
        base_length = len(self.dataset)
        if self.enable_augmentation:
            return base_length * 2
        return base_length

    def __getitem__(self, index):
        # Determine if this is an augmented sample
        is_augmented = False
        if self.enable_augmentation and index >= len(self.dataset):
            # This is an augmented sample - get the original index
            original_index = index - len(self.dataset)
            is_augmented = True
        else:
            original_index = index

        # Get the original sample
        result = self.dataset[original_index]
        
        # Apply augmentation if this is an augmented sample
        if is_augmented:
            # Extract X (first element) and apply augmentation
            X = result[0]
            X_aug = self.apply_augmentation(self.augmentation_name, X, **self.augmentation_params)
            # Replace X with augmented version
            result = (X_aug,) + result[1:]
        
        return result

def process_challenge_2_data(dataset, window_length=2.0, crop_size=2.0, sfreq=100, seed=None):
    """Process data for Challenge 2 (P-factor Prediction)."""
    # Filter out recordings that are too short or have missing p-factor
    filtered_dataset = BaseConcatDataset(
        [
            ds
            for ds in dataset.datasets
            if not ds.description.subject in PROBLEMATIC_SUBJECTS
            and ds.raw.n_times >= 4 * sfreq
            and len(ds.raw.ch_names) == 129
            and not np.isnan(ds.description["p_factor"])
        ]
    )
    
    print(f"Filtered dataset contains {len(filtered_dataset)} recordings")
    
    # Create fixed-length windows
    windows_ds = create_fixed_length_windows(
        filtered_dataset,
        window_size_samples=int(window_length * sfreq),
        window_stride_samples=int(crop_size * sfreq),
        drop_last_window=True,
    )
    
    # Wrap each sub-dataset in the windows_ds
    windows_ds = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=int(crop_size * sfreq), seed=seed) 
         for ds in windows_ds.datasets]
    )
    
    print(f"Created {len(windows_ds)} windows")
    
    return windows_ds

def split_data_by_subjects(dataset, valid_frac=0.1, test_frac=0.001, seed=2025, challenge=1):
    """Split dataset by subjects to avoid data leakage."""
    if valid_frac == 0:
        test_frac = int(2)
    if challenge == 1:
        # For Challenge 1, use metadata to get subjects
        meta_information = dataset.get_metadata()
        subjects = meta_information["subject"].unique()
    else:
        # For Challenge 2, get subjects from dataset descriptions
        subjects = []
        for ds in dataset.datasets:
            if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'description'):
                # This is a DatasetWrapper, get subject from underlying dataset
                subjects.append(ds.dataset.description["subject"])
            elif hasattr(ds, 'description'):
                # This is a regular dataset
                subjects.append(ds.description["subject"])
        subjects = list(set(subjects))
    
    # Remove problematic subjects
    subjects = [s for s in subjects if s not in PROBLEMATIC_SUBJECTS]
    
    print(f"Found {len(subjects)} subjects for splitting")
    
    # Ensure we have enough subjects for splitting
    if len(subjects) < 2:
        raise ValueError(f"Not enough subjects for splitting: {len(subjects)}. Need at least 2 subjects.")
    
    # Split subjects
    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac), 
        random_state=check_random_state(seed), shuffle=True
    )
    
    print(f"Split subjects: {len(train_subj)} train, {len(valid_test_subject)} valid+test")

    if valid_frac == 0:
        valid_subj = valid_test_subject
        test_subj = valid_test_subject
    if test_frac>0.0:
        valid_subj, test_subj = train_test_split(
            valid_test_subject, test_size=test_frac, 
            random_state=check_random_state(seed + 1), shuffle=True
        )
    else:
        valid_subj = valid_test_subject
        test_subj = valid_test_subject
    
    print(f"Final split: {len(train_subj)} train, {len(valid_subj)} valid, {len(test_subj)} test")
    
    # Create train/valid/test splits
    if challenge == 1:
        # For Challenge 1, use subject split
        subject_split = dataset.split("subject")
        train_set = []
        valid_set = []
        test_set = []
        
        for s in subject_split:
            if s in train_subj:
                train_set.append(subject_split[s])
            elif s in valid_subj:
                valid_set.append(subject_split[s])
            elif s in test_subj:
                test_set.append(subject_split[s])
        
        train_set = BaseConcatDataset(train_set)
        valid_set = BaseConcatDataset(valid_set)
        if test_frac>0.0:
            test_set = BaseConcatDataset(test_set)
        else:
            test_set = BaseConcatDataset(valid_set)
        
    else:
        # For Challenge 2, split by dataset descriptions
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        
        for ds in dataset.datasets:
            if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'description'):
                # This is a DatasetWrapper, get subject from underlying dataset
                subject = ds.dataset.description["subject"]
            elif hasattr(ds, 'description'):
                # This is a regular dataset
                subject = ds.description["subject"]
            else:
                continue
            if subject in train_subj:
                train_datasets.append(ds)
            elif subject in valid_subj:
                valid_datasets.append(ds)
            elif subject in test_subj:
                test_datasets.append(ds)
        
        train_set = BaseConcatDataset(train_datasets)
        valid_set = BaseConcatDataset(valid_datasets)
        if test_frac>0.0:
            test_set = BaseConcatDataset(test_datasets)
        else:
            test_set = BaseConcatDataset(valid_datasets)
    
    print(f"Train: {len(train_set)} examples")
    print(f"Valid: {len(valid_set)} examples")
    # print(f"Test: {len(test_set)} examples")
    
    return train_set, valid_set, test_set

def create_cv_splits(dataset, cv_folds=5, seed=2025, challenge=1):
    """Create cross-validation splits by subjects."""
    if challenge == 1:
        meta_information = dataset.get_metadata()
        subjects = meta_information["subject"].unique()
    else:
        subjects = []
        for ds in dataset.datasets:
            if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'description'):
                # This is a DatasetWrapper, get subject from underlying dataset
                subjects.append(ds.dataset.description["subject"])
            elif hasattr(ds, 'description'):
                # This is a regular dataset
                subjects.append(ds.description["subject"])
        subjects = list(set(subjects))
    
    # Remove problematic subjects
    subjects = [s for s in subjects if s not in PROBLEMATIC_SUBJECTS]
    
    # Create KFold splitter
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(subjects)):
        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        
        print(f"CV Fold {fold + 1}: Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
        
        # Create splits (simplified - same logic for both challenges)
        if challenge == 1:
            subject_split = dataset.split("subject")
            train_set = []
            val_set = []
            
            for s in subject_split:
                if s in train_subjects:
                    train_set.append(subject_split[s])
                elif s in val_subjects:
                    val_set.append(subject_split[s])
            
            train_set = BaseConcatDataset(train_set)
            val_set = BaseConcatDataset(val_set)
            
        else:
            train_datasets = []
            val_datasets = []
            
            for ds in dataset.datasets:
                if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'description'):
                    # This is a DatasetWrapper, get subject from underlying dataset
                    subject = ds.dataset.description["subject"]
                elif hasattr(ds, 'description'):
                    # This is a regular dataset
                    subject = ds.description["subject"]
                else:
                    continue
                if subject in train_subjects:
                    train_datasets.append(ds)
                elif subject in val_subjects:
                    val_datasets.append(ds)
            
            train_set = BaseConcatDataset(train_datasets)
            val_set = BaseConcatDataset(val_datasets)
        
        cv_splits.append((train_set, val_set))
    
    return cv_splits

def load_and_process_multitask_data(data_folder, task_list, data_releases, eval_release, use_all_releases,
                                  challenge, max_subjects=None, window_length=2.0, shift=0.5, crop_size=2.0,
                                  sfreq=100, valid_frac=0.1, test_frac=0.1, enable_cv=False, cv_folds=5, seed=2025,
                                  load_cached_dataset=False, cache_dataset=False, cache_dir="dataset_cache",
                                  enable_augmentation=False, augmentation_name="gaussian_noise", augmentation_params=None):
    """Load and process data from multiple tasks for pretraining."""
    
    # Generate cache key for multitask data
    cache_key = generate_cache_key(data_folder, None, data_releases, eval_release, use_all_releases,
                                 challenge, max_subjects, window_length, shift, crop_size, sfreq,
                                 valid_frac, test_frac, enable_cv, cv_folds, seed, enable_augmentation,
                                 augmentation_name, augmentation_params, task_list)
    
    # Try to load from cache first
    if load_cached_dataset:
        cached_datasets = load_dataset_cache(cache_key, cache_dir)
        if cached_datasets is not None:
            return cached_datasets
        else:
            print("Cache load failed, proceeding with normal data loading...")
    
    # Check if cache already exists when cache_dataset is requested
    if cache_dataset:
        existing_cache = load_dataset_cache(cache_key, cache_dir)
        if existing_cache is not None:
            print("WARNINGWARNINGWARNING: Cache already exists and loads successfully. Skipping cache save to prevent overwrite.")
            cache_dataset = False  # Don't overwrite existing cache
    
    all_datasets = []
    
    # Load data for each task
    all_train_datasets = []
    all_eval_datasets = []
    
    for task_name in task_list:
        print(f"Loading data for task: {task_name}", flush=True)
        
        if use_all_releases:
            # Use all releases except eval_release for training
            train_releases = [r for r in data_releases if r != eval_release]
            train_dataset = load_multiple_releases_data(data_folder, task_name, train_releases, max_subjects)
            # Load evaluation data from eval_release
            eval_dataset = load_single_release_data(data_folder, task_name, eval_release, max_subjects)
        else:
            # Single release mode
            train_dataset = load_single_release_data(data_folder, task_name, data_releases[0], max_subjects)
            eval_dataset = None
        
        # Process data based on challenge
        if challenge == 1:
            processed_train_dataset = process_challenge_1_data(train_dataset, window_length, shift, sfreq, seed=seed)
            if eval_dataset is not None:
                processed_eval_dataset = process_challenge_1_data(eval_dataset, window_length, shift, sfreq, seed=seed)
            else:
                processed_eval_dataset = None
        else:
            processed_train_dataset = process_challenge_2_data(train_dataset, window_length, crop_size, sfreq, seed=seed)
            if eval_dataset is not None:
                processed_eval_dataset = process_challenge_2_data(eval_dataset, window_length, crop_size, sfreq, seed=seed)
            else:
                processed_eval_dataset = None
        
        all_train_datasets.append(processed_train_dataset)
        if processed_eval_dataset is not None:
            all_eval_datasets.append(processed_eval_dataset)
        print(f"Loaded {len(processed_train_dataset)} training samples from {task_name}")
        if processed_eval_dataset is not None:
            print(f"Loaded {len(processed_eval_dataset)} evaluation samples from {task_name}")
    
    # Concatenate all training datasets
    combined_train_dataset = BaseConcatDataset(all_train_datasets)
    print(f"Combined training dataset contains {len(combined_train_dataset)} samples from {len(task_list)} tasks")
    
    # Concatenate evaluation datasets if available
    if all_eval_datasets:
        combined_eval_dataset = BaseConcatDataset(all_eval_datasets)
        print(f"Combined evaluation dataset contains {len(combined_eval_dataset)} samples from {len(task_list)} tasks")
    else:
        combined_eval_dataset = None
    
    # Split combined dataset by subjects
    if enable_cv:
        result = create_cv_splits(combined_train_dataset, cv_folds, seed, challenge)
    else:
        if use_all_releases:
            # For all releases, use 80:20 split
            train_split, valid_split, _ = split_data_by_subjects(combined_train_dataset, valid_frac=0.2, test_frac=0.0, seed=seed, challenge=challenge)
            result = train_split, valid_split, combined_eval_dataset
        else:
            # For single release, use 80:10:10 split
            result = split_data_by_subjects(combined_train_dataset, valid_frac=0.1, test_frac=0.1, seed=seed, challenge=challenge)
    
    # Apply augmentation to training data only
    if enable_augmentation:
        if use_all_releases:
            train_split, valid_split, eval_processed = result
            train_split = AugmentationWrapper(train_split, enable_augmentation=True,
                                            augmentation_name=augmentation_name,
                                            augmentation_params=augmentation_params)
            print(f"Applied augmentation to training data: {len(train_split)} total samples")
            result = train_split, valid_split, eval_processed
        else:
            train_split, valid_split, test_split = result
            train_split = AugmentationWrapper(train_split, enable_augmentation=True,
                                            augmentation_name=augmentation_name,
                                            augmentation_params=augmentation_params)
            print(f"Applied augmentation to training data: {len(train_split)} total samples")
            result = train_split, valid_split, test_split
    
    # Save to cache if requested
    if cache_dataset:
        save_dataset_cache(result, cache_key, cache_dir)
    
    return result

def load_and_process_data(data_folder, task_name, data_releases, eval_release, use_all_releases, 
                         challenge, max_subjects=None, window_length=2.0, shift=0.5, crop_size=2.0, 
                         sfreq=100, valid_frac=0.1, test_frac=0.1, enable_cv=False, cv_folds=5, seed=2025,
                         load_cached_dataset=False, cache_dataset=False, cache_dir="dataset_cache",
                         enable_augmentation=False, augmentation_name="gaussian_noise", 
                         augmentation_params=None, task_list=None, existing_cache_key=None):
    """Main function to load and process data for EEG Challenge."""
    
    pl.seed_everything(seed)

    # Handle multi-task loading
    if task_list is not None:
        return load_and_process_multitask_data(data_folder, task_list, data_releases, eval_release, use_all_releases,
                                             challenge, max_subjects, window_length, shift, crop_size, sfreq,
                                             valid_frac, test_frac, enable_cv, cv_folds, seed,
                                             load_cached_dataset, cache_dataset, cache_dir,
                                             enable_augmentation, augmentation_name, augmentation_params)

    # Generate cache key
    if existing_cache_key is None:
        cache_key = generate_cache_key(data_folder, task_name, data_releases, eval_release, use_all_releases,
                                    challenge, max_subjects, window_length, shift, crop_size, sfreq,
                                    valid_frac, test_frac, enable_cv, cv_folds, seed, enable_augmentation,
                                    augmentation_name, augmentation_params)
    else:
        cache_key = existing_cache_key
    
    # Try to load from cache first
    if load_cached_dataset:
        cached_datasets = load_dataset_cache(cache_key, cache_dir)
        if cached_datasets is not None:
            return cached_datasets
        else:
            print("Cache load failed, proceeding with normal data loading...", flush=True)
    
    # Check if cache already exists when cache_dataset is requested
    if cache_dataset:
        existing_cache = load_dataset_cache(cache_key, cache_dir)
        if existing_cache is not None:
            print("WARNINGWARNINGWARNING: Cache already exists and loads successfully. Skipping cache save to prevent overwrite.")
            cache_dataset = False  # Don't overwrite existing cache
    
    # Load data
    if use_all_releases:
        # Use all releases except eval_release for training
        train_releases = [r for r in data_releases if r != eval_release]
        print(f"Training on releases: {train_releases}")
        print(f"Evaluating on release: {eval_release}")
        
        # Load training data from all releases except eval_release
        train_dataset = load_multiple_releases_data(data_folder, task_name, train_releases, max_subjects)
        
        # Load evaluation data from eval_release
        eval_dataset = load_single_release_data(data_folder, task_name, eval_release, max_subjects)
        
        # Process data based on challenge
        if challenge == 1:
            train_processed = process_challenge_1_data(train_dataset, window_length, shift, sfreq, seed=seed)
            eval_processed = process_challenge_1_data(eval_dataset, window_length, shift, sfreq, seed=seed)
        else:
            train_processed = process_challenge_2_data(train_dataset, window_length, crop_size, sfreq, seed=seed)
            eval_processed = process_challenge_2_data(eval_dataset, window_length, crop_size, sfreq, seed=seed)
        
        # Split training data into train/validation (80:20 split)
        if enable_cv:
            # For CV, use all training data for cross-validation
            result = create_cv_splits(train_processed, cv_folds, seed, challenge), eval_processed
        else:
            # Split training data into train/validation (80:20)
            train_split, valid_split, _ = split_data_by_subjects(train_processed, valid_frac=0.1, test_frac=0.0, seed=seed, challenge=challenge)
            print(f"Test: {len(eval_processed)} examples")
            result = train_split, valid_split, eval_processed

        # Apply augmentation to training data only
        if enable_augmentation:
            train_split, valid_split, eval_processed = result
            train_split = AugmentationWrapper(train_split, enable_augmentation=True,
                                            augmentation_name=augmentation_name,
                                            augmentation_params=augmentation_params)
            print(f"Applied augmentation to training data: {len(train_split)} total samples")
            result = train_split, valid_split, eval_processed
        
    else:
        # Single release mode - load and split with 80:10:10
        full_dataset = load_single_release_data(data_folder, task_name, data_releases[0], max_subjects)
        
        # Process data based on challenge
        if challenge == 1:
            processed_dataset = process_challenge_1_data(full_dataset, window_length, shift, sfreq, seed=seed)
        else:
            processed_dataset = process_challenge_2_data(full_dataset, window_length, crop_size, sfreq, seed=seed)
        
        # Split data with 80:10:10
        if enable_cv:
            result = create_cv_splits(processed_dataset, cv_folds, seed, challenge)
            # For CV, apply augmentation to each fold's training data
            if enable_augmentation:
                cv_splits, eval_processed = result
                augmented_cv_splits = []
                for fold_data in cv_splits:
                    train_fold, valid_fold = fold_data
                    train_fold = AugmentationWrapper(train_fold, enable_augmentation=True,
                                                   augmentation_name=augmentation_name,
                                                   augmentation_params=augmentation_params)
                    augmented_cv_splits.append((train_fold, valid_fold))
                result = augmented_cv_splits, eval_processed
        else:
            result = split_data_by_subjects(processed_dataset, valid_frac=0.1, test_frac=0.1, seed=seed, challenge=challenge)
            # Apply augmentation to training data only (after splitting)
            if enable_augmentation:
                train_split, valid_split, test_split = result
                train_split = AugmentationWrapper(train_split, enable_augmentation=True,
                                                augmentation_name=augmentation_name,
                                                augmentation_params=augmentation_params)
                print(f"Applied augmentation to training data: {len(train_split)} total samples")
                result = train_split, valid_split, test_split
    
    # Save to cache if requested
    if cache_dataset:
        save_dataset_cache(result, cache_key, cache_dir)
    
    return result

if __name__ == "__main__":
    print("Testing data loading...")
    try:
        dataset = load_single_release_data("local_R1", "contrastChangeDetection", "R1_L100_bdf")
        print(f"Loaded dataset with {len(dataset)} recordings")
    except Exception as e:
        print(f"Error loading data: {e}")


class SimpleWrapper:
    """Robust data wrapper with per-channel z-scoring for EEG data."""
    
    def __init__(self, dataset, scale_factor=1e3, normalize=False, stats_n_samples=200, stats_seed=42):
        self.dataset = dataset
        self.scale_factor = float(scale_factor)
        self.normalize = normalize
        self.stats_n_samples = int(stats_n_samples)
        self.stats_seed = int(stats_seed)

        # Prepare placeholders (per-channel); must be set when normalize True
        self.global_mean = None
        self.global_std = None

        # Compute normalization stats from training data only (call before DataLoader workers)
        if self.normalize:
            self._compute_stats()

    def _compute_stats(self):
        """
        Compute per-channel mean and std using an online algorithm over a random sample of dataset items.
        This returns self.global_mean (shape C,) and self.global_std (shape C,).
        """
        import math
        
        # random sample of indices (without replacement), robust to small datasets
        n = len(self.dataset)
        k = min(self.stats_n_samples, n)
        rng = random.Random(self.stats_seed)
        indices = rng.sample(range(n), k) if k < n else list(range(n))

        # initialize accumulators for online mean/var (per channel)
        chan_mean = None
        chan_M2 = None
        count = 0

        for idx in indices:
            sample = self.dataset[idx]
            data = sample[0] if isinstance(sample, (list, tuple)) else sample
            # convert to torch float32 on CPU (no device)
            if not torch.is_tensor(data):
                data = torch.tensor(data, dtype=torch.float32)
            # scale to mV
            data = data * self.scale_factor   # data shape (C, T) or (C,T,...)
            # reduce to (C, *) flatten time axis(s)
            C = data.shape[0]
            data_flat = data.view(C, -1).float()  # (C, Ntime)
            # compute per-channel mean for this sample
            sample_mean = data_flat.mean(dim=1)   # (C,)
            sample_var = data_flat.var(dim=1, unbiased=False)  # (C,)
            sample_count = data_flat.shape[1]

            # online update per-channel using Welford-style for sample-blocks
            if chan_mean is None:
                chan_mean = sample_mean.clone()
                # convert var to M2 = var * (n)
                chan_M2 = sample_var * sample_count
                count = sample_count
            else:
                # combine previous (mean1,M21,count1) with new block (mean2,M22,count2)
                mean1, M2_1, n1 = chan_mean, chan_M2, count
                mean2, M2_2, n2 = sample_mean, sample_var * sample_count, sample_count
                delta = mean2 - mean1
                new_n = n1 + n2
                chan_mean = mean1 + delta * (n2 / new_n)
                chan_M2 = M2_1 + M2_2 + (delta ** 2) * (n1 * n2 / new_n)
                count = new_n

        # finalize
        if chan_mean is None:
            raise RuntimeError("No data found to compute stats.")
        # population variance per channel
        chan_var = chan_M2 / count
        chan_std = torch.sqrt(chan_var + 1e-12)

        self.global_mean = chan_mean.numpy()   # convert to numpy scalar array shape (C,)
        self.global_std = chan_std.numpy()
        print(f"Computed per-channel normalization stats: mean shape {self.global_mean.shape}, std shape {self.global_std.shape}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, (list, tuple)):
            data = sample[0]
        else:
            data = sample

        # Convert to tensor and scale
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        data = data * self.scale_factor  # scale to mV

        # Apply per-channel normalization if enabled (broadcast across time)
        if self.normalize:
            if self.global_mean is None or self.global_std is None:
                raise RuntimeError("Normalization stats missing; ensure _compute_stats() ran on training set.")
            # convert stats to torch on same device (cpu)
            gm = torch.as_tensor(self.global_mean, dtype=torch.float32).view(-1, 1)
            gs = torch.as_tensor(self.global_std, dtype=torch.float32).view(-1, 1)
            data = (data - gm) / (gs + 1e-8)

        if isinstance(sample, (list, tuple)):
            # return (data, ...) preserving metadata/labels
            return (data,) + tuple(sample[1:])
        return data
