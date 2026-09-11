import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .generic import GenericModel
from .load import get_model


class EnsemblingModel(GenericModel):
    """
    Ensemble multiple trained models with various aggregation strategies.

    Supported methods:
    - 'mean': Simple average of predictions
    - 'median': Robust median aggregation
    - 'weighted': Fixed weights (e.g., based on validation performance)
    - 'learnable': Learnable weights optimized during training (treats ensemble weights as trainable parameters that are optimized during training, rather than being fixed
  based on validation performance.)
    - 'inverse_loss': Weights based on inverse validation loss (softmax normalized) -> **recommended**

    Args:
        models: List of trained model instances (nn.Module)
        method: Aggregation method ('mean', 'median', 'weighted', 'learnable', 'inverse_loss')
        weights: Optional weights for 'weighted' method (Tensor, list, or dict)
        model_names: Optional list of model names for logging
        **kwargs: Additional arguments passed to GenericModel

    Example:
        # Load trained models
        model1 = get_model('EEGNeX', n_chans=129, n_outputs=1, n_times=200, sfreq=100)
        model2 = get_model('BIOT', n_chans=129, n_outputs=1, n_times=200, sfreq=100)
        model3 = get_model('EEGConformer', n_chans=129, n_outputs=1, n_times=200, sfreq=100)

        # Create ensemble with mean averaging
        ensemble = EnsemblingModel(
            models=[model1, model2, model3],
            method='mean',
            loss=torch.nn.MSELoss(),
            optimizer='adamw',
            optimizer_params={'lr': 1e-4}
        )

        # Or with validation-based weights
        val_nrmse = {'EEGNeX': 0.15, 'BIOT': 0.18, 'EEGConformer': 0.16}
        ensemble = EnsemblingModel(
            models=[model1, model2, model3],
            model_names=['EEGNeX', 'BIOT', 'EEGConformer'],
            method='inverse_loss',
            weights=val_nrmse,
            loss=torch.nn.MSELoss()
        )
    """

    def __init__(self, models, method='mean', weights=None, model_names=None, **kwargs):
        # Initialize parent class with a dummy model (we'll override forward)
        super().__init__(model=nn.Identity(), **kwargs)
        # Store models in ModuleList
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.method = method
        self.model_names = model_names or [f'model_{i}' for i in range(self.num_models)]

        # Validate inputs
        if self.num_models == 0:
            raise ValueError("Must provide at least one model")

        # Initialize method-specific parameters
        if method == 'weighted':
            if weights is None:
                raise ValueError("Must provide weights for 'weighted' method")
            self._init_fixed_weights(weights)

        elif method == 'learnable':
            # Initialize learnable weights (will be optimized during training)
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)

        elif method == 'inverse_loss':
            if weights is None:
                raise ValueError("Must provide validation losses/metrics for 'inverse_loss' method")
            self._init_inverse_loss_weights(weights)

        elif method in ['mean', 'median']:
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unknown method: {method}. Choose from ['mean', 'median', 'weighted', 'learnable', 'inverse_loss']")

        # Set all ensemble models to eval mode by default
        for model in self.models:
            model.eval()

    def _init_fixed_weights(self, weights):
        """Initialize fixed weights from dict, list, or tensor."""
        if isinstance(weights, dict):
            # Weights provided as {model_name: weight}
            weight_list = []
            for name in self.model_names:
                if name not in weights:
                    raise ValueError(f"Model '{name}' not found in weights dict")
                weight_list.append(weights[name])
            weights_tensor = torch.tensor(weight_list, dtype=torch.float32)
        elif isinstance(weights, (list, tuple)):
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
        elif isinstance(weights, torch.Tensor):
            weights_tensor = weights.float()
        else:
            raise ValueError("weights must be dict, list, tuple, or Tensor")

        if len(weights_tensor) != self.num_models:
            raise ValueError(f"Number of weights ({len(weights_tensor)}) must match number of models ({self.num_models})")

        # Normalize weights to sum to 1
        weights_tensor = weights_tensor / weights_tensor.sum()
        self.register_buffer('weights', weights_tensor)
        print(f"Fixed weights: {dict(zip(self.model_names, weights_tensor.tolist()))}")

    def _init_inverse_loss_weights(self, val_metrics):
        """
        Initialize weights based on inverse validation loss/error.
        Lower loss = higher weight.

        Args:
            val_metrics: Dict {model_name: loss} or list of losses
        """
        if isinstance(val_metrics, dict):
            # Metrics provided as {model_name: metric_value}
            metric_list = []
            for name in self.model_names:
                if name not in val_metrics:
                    raise ValueError(f"Model '{name}' not found in val_metrics dict")
                metric_list.append(val_metrics[name])
            metrics_tensor = torch.tensor(metric_list, dtype=torch.float32)
        elif isinstance(val_metrics, (list, tuple)):
            metrics_tensor = torch.tensor(val_metrics, dtype=torch.float32)
        elif isinstance(val_metrics, torch.Tensor):
            metrics_tensor = val_metrics.float()
        else:
            raise ValueError("val_metrics must be dict, list, tuple, or Tensor")

        # Compute weights as softmax of inverse losses
        # Use negative log to convert to "scores" (lower loss = higher score)
        inv_metrics = -torch.log(metrics_tensor + 1e-8)
        weights = F.softmax(inv_metrics, dim=0)

        self.register_buffer('weights', weights)
        self.register_buffer('val_metrics', metrics_tensor)

        print(f"Inverse loss weights (val metrics: {dict(zip(self.model_names, metrics_tensor.tolist()))}):")
        for name, weight in zip(self.model_names, weights.tolist()):
            print(f"  {name}: {weight:.4f}")

    def forward(self, x):
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor or tuple of tensors

        Returns:
            Aggregated predictions
        """
        # Collect predictions from all models
        predictions = []
        for model in self.models:
            if isinstance(x, (list, tuple)):
                pred = model(*x)
            else:
                pred = model(x)
            predictions.append(pred)

        # Stack predictions: [num_models, batch_size, ...]
        predictions = torch.stack(predictions, dim=0)

        # Apply aggregation method
        if self.method == 'mean':
            return predictions.mean(dim=0)

        elif self.method == 'median':
            return predictions.median(dim=0)[0]

        elif self.method in ['weighted', 'inverse_loss']:
            # Reshape weights for broadcasting: [num_models, 1, 1, ...]
            weight_shape = [self.num_models] + [1] * (predictions.dim() - 1)
            weights = self.weights.view(*weight_shape)
            return (weights * predictions).sum(dim=0)

        elif self.method == 'learnable':
            # Softmax to ensure weights sum to 1
            weights = F.softmax(self.weights, dim=0)
            weight_shape = [self.num_models] + [1] * (predictions.dim() - 1)
            weights = weights.view(*weight_shape)
            return (weights * predictions).sum(dim=0)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_weights(self):
        """Get current ensemble weights."""
        if self.method == 'learnable':
            return F.softmax(self.weights, dim=0).detach().cpu().numpy()
        elif self.method in ['weighted', 'inverse_loss']:
            return self.weights.detach().cpu().numpy()
        else:
            return np.ones(self.num_models) / self.num_models

    def print_weights(self):
        """Print current ensemble weights."""
        weights = self.get_weights()
        print("\nEnsemble weights:")
        for name, weight in zip(self.model_names, weights):
            print(f"  {name}: {weight:.4f}")


def load_models_from_checkpoints(model_configs, checkpoint_paths, device='cpu'):
    """
    Load multiple trained models from checkpoint files.

    Args:
        model_configs: List of dicts with model configuration
            Each dict should contain: 'model_name', 'n_chans', 'n_outputs', 'n_times', 'sfreq'
        checkpoint_paths: List of paths to .pt or .ckpt files
        device: Device to load models on ('cpu', 'cuda', etc.)

    Returns:
        List of loaded models in eval mode

    Example:
        configs = [
            {'model_name': 'EEGNeX', 'n_chans': 129, 'n_outputs': 1, 'n_times': 200, 'sfreq': 100},
            {'model_name': 'BIOT', 'n_chans': 129, 'n_outputs': 1, 'n_times': 200, 'sfreq': 100},
        ]
        paths = [
            'results/stats_EEGNeX/model/weights_challenge_1_EEGNeX.pt',
            'results/stats_BIOT/model/weights_challenge_1_BIOT.pt',
        ]
        models = load_models_from_checkpoints(configs, paths)
    """
    if len(model_configs) != len(checkpoint_paths):
        raise ValueError("Number of configs must match number of checkpoint paths")

    models = []
    for config, path in zip(model_configs, checkpoint_paths):
        print(f"Loading {config['model_name']} from {path}")

        # Create model architecture
        model = get_model(
            model_name=config['model_name'],
            n_chans=config['n_chans'],
            n_outputs=config['n_outputs'],
            n_times=config['n_times'],
            sfreq=config.get('sfreq', 100)
        )

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present (from GenericModel wrapper)
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()
                             if k.startswith('model.')}
                if not state_dict:
                    # No 'model.' prefix, use full state dict
                    state_dict = checkpoint['state_dict']
            else:
                # Regular state dict
                state_dict = checkpoint
        else:
            # Direct state dict
            state_dict = checkpoint

        # Load weights
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Could not load with strict=True, trying strict=False")
            print(f"Error: {e}")
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Successfully loaded {config['model_name']}")

    return models


def find_best_checkpoints(results_folder, model_names, pattern='weights_challenge_1_{model_name}.pt'):
    """
    Find checkpoint files for multiple models.

    Args:
        results_folder: Base results folder path
        model_names: List of model names to find
        pattern: Checkpoint filename pattern (use {model_name} placeholder)

    Returns:
        List of checkpoint paths

    Example:
        paths = find_best_checkpoints(
            'results',
            ['EEGNeX', 'BIOT', 'EEGConformer'],
            pattern='weights_challenge_1_{model_name}.pt'
        )
    """
    results_path = Path(results_folder)
    checkpoint_paths = []

    for model_name in model_names:
        # Try pattern with model name
        model_dir = results_path / f"stats_{model_name}" / "model"
        checkpoint_file = model_dir / pattern.format(model_name=model_name)

        if checkpoint_file.exists():
            checkpoint_paths.append(str(checkpoint_file))
        else:
            # Try to find any .pt or .ckpt file in model directory
            pt_files = list(model_dir.glob("*.pt"))
            ckpt_files = list(model_dir.glob("*_best*.ckpt"))

            if pt_files:
                checkpoint_paths.append(str(pt_files[0]))
            elif ckpt_files:
                checkpoint_paths.append(str(ckpt_files[0]))
            else:
                raise FileNotFoundError(f"No checkpoint found for {model_name} in {model_dir}")

    return checkpoint_paths


def extract_val_metrics(results_folder, model_names, metric='val_nrmse', return_dict=True):
    """
    Extract validation metrics from training logs for ensemble weighting.

    Args:
        results_folder: Base results folder path
        model_names: List of model names to extract metrics for
        metric: Metric to extract ('val_nrmse', 'val_loss')
        return_dict: If True, return dict {model_name: metric}, else return list of metrics

    Returns:
        Dict mapping model names to validation metrics, or list of metrics

    Example:
        # Get validation NRMSE for ensemble weighting
        val_nrmse = extract_val_metrics(
            'results/braindecode/challenge_1',
            ['EEGNeX', 'BIOT', 'EEGConformer'],
            metric='val_nrmse'
        )
        # Returns: {'EEGNeX': 0.15, 'BIOT': 0.18, 'EEGConformer': 0.16}

        # Use with EnsemblingModel
        ensemble = EnsemblingModel(
            models=[model1, model2, model3],
            model_names=['EEGNeX', 'BIOT', 'EEGConformer'],
            method='inverse_loss',
            weights=val_nrmse
        )
    """
    import pandas as pd
    import glob

    results_path = Path(results_folder)
    metrics_dict = {}

    for model_name in model_names:
        print(f"Extracting {metric} for {model_name}...")

        # Find the metrics CSV file
        model_dir = results_path / f"stats_{model_name}" / "model" / "log" / "csv"

        # Look for supervised training logs
        csv_pattern = model_dir / f"{model_name}_supervised*" / "version_*" / "metrics.csv"
        csv_files = glob.glob(str(csv_pattern))

        if not csv_files:
            print(f"  Warning: No metrics.csv found for {model_name}, skipping...")
            continue

        # Get the latest version (highest version number)
        latest_csv = max(csv_files, key=lambda x: int(Path(x).parent.name.split('_')[-1]))
        print(f"  Found: {latest_csv}")

        # Read CSV and extract metric
        try:
            df = pd.read_csv(latest_csv)

            if metric not in df.columns:
                # Try to compute val_nrmse if not present
                if metric == 'val_nrmse' and 'val_loss' in df.columns:
                    print(f"  Warning: {metric} not found, using val_loss instead")
                    metric_col = 'val_loss'
                else:
                    print(f"  Warning: {metric} not found in CSV for {model_name}")
                    continue
            else:
                metric_col = metric

            # Remove NaN values and get best (minimum) metric
            valid_df = df[df[metric_col].notna()]
            if len(valid_df) == 0:
                print(f"  Warning: No valid {metric_col} values for {model_name}")
                continue

            best_metric = valid_df[metric_col].min()
            metrics_dict[model_name] = float(best_metric)

            print(f"  {metric_col}: {best_metric:.6f}")

        except Exception as e:
            print(f"  Error reading CSV for {model_name}: {e}")
            continue

    if not metrics_dict:
        raise RuntimeError("No validation metrics found for any model")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 60)
    for name, val in metrics_dict.items():
        print(f"{name:20s}: {val:.6f}")
    print("=" * 60)

    if return_dict:
        return metrics_dict
    else:
        # Return list in same order as model_names
        return [metrics_dict[name] for name in model_names if name in metrics_dict]
