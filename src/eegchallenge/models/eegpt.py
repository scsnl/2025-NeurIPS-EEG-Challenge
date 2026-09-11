import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_eegpt_classifier():
    """Import ``EEGPTClassifier`` from the upstream EEGPT repository.

    EEGPT is a third-party model and is not vendored here. Clone it and put it
    on ``PYTHONPATH`` to use the ``EEGPT`` backbone:

        git clone https://github.com/BINE022/EEGPT
        export PYTHONPATH=/path/to/EEGPT/downstream:$PYTHONPATH

    Deferred to call time so that importing this package does not require
    EEGPT — every other model works without it.
    """
    try:
        from Modules.models.EEGPT_mcae_finetune import EEGPTClassifier
    except ImportError as exc:  # pragma: no cover - depends on external repo
        raise ImportError(
            "The EEGPT backbone requires the upstream EEGPT repository on "
            "PYTHONPATH (https://github.com/BINE022/EEGPT). See "
            "eegchallenge/models/eegpt.py for setup, or choose another model."
        ) from exc
    return EEGPTClassifier

class EEGPTWrapper(nn.Module):
    """
    Wrapper to make EEGPT compatible with our framework interface.

    Inputs (Inherited from braindecode):
        - n_chans: Number of EEG channels.
        - n_outputs: Number of outputs of the model. This is the number of classes in the case of classification.
        - n_times: Number of time samples of the input window.
        - sfreq: sampling frequency.
        - pretrained_path: Path to pretrained model checkpoints
        - freeze_encoder: True = The encoder weights don't update during training. Only the classification head (final layers) gets trained, i.e., 'linear probing'. False = Fine tuning.

    Inputs of EEGPT model (Original classifier):
        - num_classes: Number of outputs (e.g., 1 for regression)
        - in_channels: Your actual number of EEG channels (e.g., 129)
        - img_size: [target_channels, target_time_len] - adjusted dimensions
        - use_channels_names: The channel name list you provide. It adds a learnable spatial filter that transforms input channels to match EEGPT's expected channel layout using convolutional layers. 
        - use_chan_conv: Set to True in your wrapper
        - use_avg = False: Do not subtracts the mean across channels (dim=-2) before interpolation. No DC offset removal or baseline correction.

    # Architecture: Input [batch, 129, 200] → Interpolate → [batch, 129, 512] 
    # → chan_conv → [batch, 58, 512] → Patch Embed → [batch, 8, 58, 512] 
    # → Transformer → [batch, 464, 512] → Flatten → [batch, 237568] → Output Head → [batch, outputs]
    """

    def __init__(self, n_chans, n_outputs, n_times, sfreq=100, pretrained_path=None, freeze_encoder=False, **kwargs):
        
        super().__init__() # calls the __init__() method of the parent class (nn.Module from PyTorch)

        self.n_chans = n_chans 
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.freeze_encoder = freeze_encoder
        self.target_channels = self._get_channel_names() # Define target channels
        self.target_time_len = 512 # target number of time points inside each segment

        print(f"EEGPT Config: {n_chans} channels -> 58 EEGPT dimensions, {n_times} -> {self.target_time_len} time points")

        # Create EEGPT model
        EEGPTClassifier = _import_eegpt_classifier()
        self.eegpt_model = EEGPTClassifier(num_classes=n_outputs, in_channels=n_chans, img_size=[len(self.target_channels), self.target_time_len], use_channels_names=self._get_channel_names(), use_chan_conv=True, use_freeze_encoder=freeze_encoder, patch_size=64, patch_stride=64, desired_time_len=self.target_time_len, **kwargs)

        # Load pretrained weights. The checkpoint is not redistributed with this
        # repository; see README "External model weights" for where to get it.
        if not pretrained_path:
            pretrained_path = os.getenv('EEGPT_CHECKPOINT')
        if not pretrained_path:
            raise RuntimeError(
                'No EEGPT checkpoint given. Pass pretrained_path=... or set '
                'EEGPT_CHECKPOINT to eegpt_mcae_58chs_4s_large4E.ckpt.'
            )
        self._load_pretrained_weights(pretrained_path)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _get_channel_names(self):
        """
        Get all 58 EEGPT channel names copied from the original EEGPT scripts.
        These are the standard EEG positions that EEGPT was pretrained on.
        chan_conv will learn to map your input channels to these 58 dimensions.
        """
        return [
            'FP1', 'FPZ', 'FP2',  # Frontal polar
            'AF3', 'AF4',          # Anterior frontal
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',  # Frontal
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',  # Frontocentral
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',  # Central
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',  # Centroparietal
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',  # Parietal
            'PO7', 'PO3', 'POZ', 'PO4', 'PO8',  # Parieto-occipital
            'O1', 'OZ', 'O2'  # Occipital
        ]

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained EEGPT weights."""
        try:
            print(f"Loading pretrained weights from {pretrained_path}")
            # For PyTorch 2.6+, we need to set weights_only=False for older checkpoints
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (for DataParallel models)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                new_state_dict[k] = v

            # Load weights (strict=False to handle size mismatches)
            missing_keys, unexpected_keys = self.eegpt_model.load_state_dict(new_state_dict, strict=False)

            print(f"Loaded pretrained weights: {len(new_state_dict)} parameters")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} (expected for different dataset configurations)")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from scratch...")


    def _freeze_encoder(self):
        """Freeze encoder parameters for linear probing."""
        print("Freezing EEGPT encoder parameters...")
        frozen_params = 0
        total_params = 0

        for name, param in self.eegpt_model.named_parameters():
            total_params += 1
            if 'target_encoder' in name or 'reconstructor' in name or 'predictor' in name:
                param.requires_grad = False
                frozen_params += 1

        print(f"Frozen {frozen_params}/{total_params} parameters")

    def forward(self, x):
        """Forward pass compatible with framework interface."""
        # x shape: (batch, channels, time)

        # Ensure consistent dtype to avoid Half/Float mismatch under autocast
        if x.dtype != torch.float16: x.to(torch.float16)

        # Check input shape
        print(f"Input shape: {x.shape}")

        if x.is_cuda:
            # Use new torch.amp API if available
            with torch.amp.autocast('cuda', enabled=False):
                output = self.eegpt_model(x)
        else:
            output = self.eegpt_model(x)

        return output

if __name__ == "__main__":
    model = EEGPTWrapper(n_chans=129, n_outputs=1, n_times=200, sfreq=100)


# LJ Notes: 
# channel_mapping: 
# EEGPT was pretrained on 58 channels (provided in the EEGPT_mcae_finetune.py file as well as copied here). However our HBN data has 129 channels.
# EEGPT provides a chan_conv() method, that automatically learns to map our input channels (e.g., 129 HBN) → 58 EEGPT dimensions. (need fine-tuning using our dataset).
# This function requires us to provide target channel names (58 channels), to map chan_conv output dimensions to correct embeddings.

# target_time_len:
# Temporal resampling is necessary because EEGPT was pretrained on 4-s EEG segments at 256 Hz, so each training sample = 4 sec × 256 Hz = 1024 time points. Our HBN has 2s EEG segments at 100 Hz = 200 time points. 
# Once target time length is defined here, it will pass to forward() for temporal interpolation inside EEGPTClassifier
# EEGPT expects exactly 512 time points to produce 8 patches with patch_size=64 (was pretrained on 4-second segments at 256Hz)
# This ensures (512 - 64) // 64 + 1 = 8 patches


