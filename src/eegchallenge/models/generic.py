import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import functional as FM
import os
from collections import OrderedDict
from torch.utils.data import Dataset

class GenericModel(pl.LightningModule):
    def __init__(self, model, loss = nn.CrossEntropyLoss(), loss_params = {}, optimizer = 'adam', optimizer_params = {}, dtype = torch.float32, use_se = False, use_cdrop = False, map_location=None, metrics = {}, show = None, average = True, show_when = ['on_epoch'], scheduler = None, scheduler_params = {}):
        super(GenericModel, self).__init__()
        self.model = model.to(dtype)
        self.loss = loss
        self.loss_params = loss_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.metrics = nn.ModuleDict(metrics)
        self.show = {m: False for m in self.metrics}
        self.show['loss'] = True
        if isinstance(show, list):
            self.show.update({k:True for k in show})
        elif isinstance(show, dict):
            self.show.update(show)
        elif show == 'all':
            self.show.update({k:True for k in metrics.keys()})
        self.average = average
        self.show_when = show_when
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.to(dtype)
        self.use_se = use_se
        self.use_cdrop = use_cdrop

        self.channel_drop = nn.Sequential(
            nn.Identity() if not use_cdrop else nn.Dropout1d(p=0.2))
        
        # Initialize SEBlock placeholder - will be set when we know input channels
        self._se = None

    def forward(self, x):
        if type(x) is list or type(x) is tuple:
            return self.model(*x)
        else:
            # Ensure consistent shape: [B, C, T]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
            elif x.dim() == 4:
                x = x.squeeze(1)    # [B, 1, C, T] -> [B, C, T]
            
            x = self.channel_drop(x)
            # Initialize SEBlock on first forward pass
            if self.use_se and self._se is None:
                self._se = SEBlock1d(x.shape[1]).to(x.device, dtype=x.dtype)
            
            if self.use_se:
                x = self._se(x)
            
            return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, log_dict = self.log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, log_dict = self.log_dict, pre = 'val_',)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, log_dict = self.log_dict, pre = 'test_')
        return metrics

    def _shared_eval_step(self, batch, batch_idx, log_dict, pre = ''):
        x = batch[0]
        y = batch[1].to(self.device, dtype = self.dtype)
        
        # Ensure y has the same shape as y_hat for proper loss calculation
        y_hat = self(x)
        if y_hat.shape != y.shape:
            y = y.unsqueeze(-1) if len(y.shape) == 1 else y
        
        _metrics = {f'{pre}{name}': m(y_hat, y) for name,m in self.metrics.items()}
        loss = _metrics[f'{pre}loss'] = self.loss(y_hat, y, **self.loss_params)
        if self.average:
            metrics = {name: torch.mean(m) for name,m in _metrics.items()}
        else:
            metrics = {name: m for name,m in _metrics.items() if len(m.shape)==0}
            for name,m in _metrics.items():
                if len(m.shape)==1:
                    for i in range(len(m)):
                        metrics[f'{name}_{i}'] = m[i]
        
        # Get batch size for logging
        batch_size = x.shape[0]
        
        log_dict({k:v for k,v in metrics.items() if self.show[k[len(pre):]]}, on_step='on_step' in self.show_when, on_epoch='on_epoch' in self.show_when, sync_dist=True, prog_bar = True, batch_size=batch_size)
        log_dict({k:v for k,v in metrics.items() if not self.show[k[len(pre):]]}, on_step='on_step' in self.show_when, on_epoch='on_epoch' in self.show_when, sync_dist=True, prog_bar = False, batch_size=batch_size)
        return loss, metrics

    def configure_optimizers(self):
        if not 'params' in self.optimizer_params.keys() or len(list(self.optimizer_params['params'])) == 0:
            self.optimizer_params['params'] = self.parameters()
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(**self.optimizer_params)
            elif self.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(**self.optimizer_params)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(**self.optimizer_params)
            elif self.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(**self.optimizer_params)
            else:
                raise NameError(f'Unknown optimizer {self.optimizer}')
        else:
            optimizer = self.optimizer(**self.optimizer_params)
        if self.scheduler is None:
            return optimizer
        else:
            return [optimizer], [self.scheduler(optimizer, self.scheduler_params)]

class MultiTaskModel(GenericModel):
    def __init__(self, model: nn.Sequential, decoders: dict, remove_layers = ['decoder'], training_policy = None, step_policy = 'alternate', **params):
        super(MultiTaskModel, self).__init__(model)
        self.models = nn.ModuleDict()
        for task,decoder in decoders.items():
            layers = OrderedDict(model.named_children())
            for l in remove_layers:
                del layers[l]
            layers.update(OrderedDict([('decoder', decoder)]))
            _params = {k: p[task] if isinstance(p, dict) and task in p else p for k,p in params.items()}
            self.models[task] = GenericModel(nn.Sequential(layers), **_params)
        self.tasks = list(decoders.keys())
        self.automatic_optimization=False
        self.training_policy = training_policy

    def __getitem__(self, task):     
        return self.models[task]

    def forward(self, x, task):
        if type(x) is list or type(x) is tuple:
            return self.model[task](*x)
        else:
            return self.model[task](x)

    def training_step(self, batch, batch_idx):
        optimizers = {task: opt for task, opt in zip(self.tasks, self.optimizers())}
        if self.training_policy is None:
            tasks = self.tasks
        elif callable(self.training_policy):
            tasks = self.training_policy(self.tasks, self.current_epoch, batch_idx)
        else:
            raise NameError('Training policy not implemented yet')
        losses = []
        for task in tasks:
            if task in batch:
                self.toggle_optimizer(optimizers[task], 0)
                loss, metrics = self.models[task]._shared_eval_step(batch[task], batch_idx, pre = f'{task}_', log_dict = self.log_dict)
                optimizers[task].zero_grad()
                self.manual_backward(loss)
                optimizers[task].step()
                self.untoggle_optimizer(optimizers[task])
                losses.append(loss)
        return torch.mean(torch.stack(losses), dim=0)

    def _shared_eval_step(self, batch, batch_idx, pre = ''):
        losses = []
        metrics = {}
        for task in batch:
            loss, _metrics = self.models[task]._shared_eval_step(batch[task], batch_idx, pre = f'{task}_{pre}_', log_dict = self.log_dict)
            losses.append(loss)
            metrics.update(_metrics)
        return torch.mean(torch.stack(losses), dim=0), metrics

    def configure_optimizers(self):
        optimizers = []
        for task in self.tasks:
            optimizers.append(self.models[task].configure_optimizers())
        if self.scheduler is None:
            return optimizers
        else:
            return optimizers, [self.scheduler(optimizer, self.scheduler_params) for optimizer in optimizers] # Debug scheduler part


class SEBlock1d(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels//r, 1, bias=False)
        self.elu = nn.ELU()
        self.fc2 = nn.Conv1d(channels//r, channels, 1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = x.mean(-1, keepdim=True)
        y = self.elu(self.fc1(y))
        y = self.sig(self.fc2(y))
        return x * y

# MAE-style Pretraining Classes
class PatchMasking:
    """Patch-based masking for EEG time series (always returns a mask)."""

    def __init__(self, patch_size: int, mask_ratio: float = 0.5, mask_seed: int = None, per_sample: bool = True):
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.mask_seed = mask_seed
        self.per_sample = per_sample

    def __call__(self, x: torch.Tensor, training: bool = True):
        """
        x: (B, C, T)
        Returns: masked_x (B,C,T), mask (B,C,T) boolean where True==masked
        """
        B, C, T = x.shape
        num_patches = max(1, T // self.patch_size)

        # compute number to mask (at least 1, at most num_patches)
        num_masked = max(1, int(round(num_patches * self.mask_ratio)))
        num_masked = min(num_masked, num_patches)

        # create per-sample RNG (deterministic if mask_seed provided)
        rng = None
        if self.mask_seed is not None:
            # use torch.Generator to avoid global seed side-effects
            rng = torch.Generator(device=x.device)
            # make seed depend on training/eval so eval masks can be stable
            rng.manual_seed(self.mask_seed + (0 if training else 1))

        # build full mask
        mask = torch.zeros((B, C, T), dtype=torch.bool, device=x.device)

        for b in range(B):
            if self.per_sample:
                perm = torch.randperm(num_patches, generator=rng, device=x.device) if rng is not None else torch.randperm(num_patches, device=x.device)
            else:
                # share same mask across samples in batch
                if b == 0:
                    shared_perm = torch.randperm(num_patches, generator=rng, device=x.device) if rng is not None else torch.randperm(num_patches, device=x.device)
                perm = shared_perm
            masked_idx = perm[:num_masked]
            for i in masked_idx:
                start = int(i) * self.patch_size
                end = min((int(i) + 1) * self.patch_size, T)
                mask[b, :, start:end] = True

        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask



class ConvDecoder(nn.Module):
    """MAE-style convolutional decoder for EEG reconstruction."""
    
    def __init__(self, encoder_feat_dim: int, output_channels: int, output_length: int):
        super().__init__()
        self.encoder_feat_dim = encoder_feat_dim
        self.output_channels = output_channels
        self.output_length = output_length
        
        # Calculate required upsampling ratio
        # Assume encoder features are ~1/4 of input length (common in EEG models)
        self.upsample_ratio = 4
        
        # MAE-style decoder: simple upsampling to exact output length
        self.decoder = nn.Sequential(
            # First upsampling: 1/4 -> 1/2
            nn.ConvTranspose1d(encoder_feat_dim, encoder_feat_dim // 2, 
                              kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Second upsampling: 1/2 -> 1/1 (full length)
            nn.ConvTranspose1d(encoder_feat_dim // 2, output_channels, 
                              kernel_size=4, stride=2, padding=1),
        )
        
        # Final projection to exact output length
        self.final_proj = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        
        # Linear projection for 2D features (EEGNeX, BIOT)
        self.linear_proj = nn.Linear(encoder_feat_dim, output_channels * output_length)
        
        # Initialize weights following MAE-EEG paper
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights following MAE-EEG methodology."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                # Use He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Weights initialized
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstructed signal with exact length matching."""
        # Handle different input shapes for MAE-EEG reconstruction
        if len(x.shape) == 3 and x.shape[-1] == 1:
            # 2D features expanded to 3D: [B, feat_dim, 1]
            # Use linear projection first, then convolution refinement
            B, feat_dim, _ = x.shape
            x = x.view(B, feat_dim)  # [B, feat_dim]
            
            # Linear projection to output size (MAE-EEG approach)
            x = self.linear_proj(x)  # [B, output_channels * output_length]
            x = x.view(B, self.output_channels, self.output_length)
            
            # Apply final convolution for refinement
            x = self.final_proj(x)
        else:
            # Standard 3D features: [B, C, T]
            # Upsample to approximately correct length
            x = self.decoder(x)
            
            # Final projection
            x = self.final_proj(x)
            
            # Ensure exact output length using adaptive pooling (MAE-EEG approach)
            if x.shape[-1] != self.output_length:
                x = F.adaptive_avg_pool1d(x, self.output_length)
        
        return x




class TransformerDecoder(nn.Module):
    """MAE-style transformer decoder for EEG reconstruction."""
    
    def __init__(self, encoder_feat_dim: int, output_channels: int, output_length: int, 
                 patch_size: int, num_layers: int = 2, num_heads: int = 3):
        super().__init__()
        self.encoder_feat_dim = encoder_feat_dim
        self.output_channels = output_channels
        self.output_length = output_length
        self.patch_size = patch_size
        
        # Patch embedding (MAE-EEG style)
        self.patch_embed = nn.Linear(encoder_feat_dim, encoder_feat_dim)
        
        # Learnable mask tokens (MAE approach)
        self.mask_token = nn.Parameter(torch.randn(encoder_feat_dim))
        
        # Transformer decoder (simplified for EEG)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=encoder_feat_dim,
            nhead=num_heads,
            dim_feedforward=encoder_feat_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to exact length
        self.output_proj = nn.Linear(encoder_feat_dim, patch_size * output_channels)
        
        # Initialize weights following MAE-EEG
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer decoder weights following MAE-EEG."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        # Transformer weights initialized
        
    def forward(self, encoder_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstructed signal with exact length matching."""
        B, C, T = mask.shape
        
        # Handle different encoder feature shapes
        if len(encoder_features.shape) == 3 and encoder_features.shape[-1] == 1:
            # 2D features expanded to 3D: [B, feat_dim, 1]
            # Use linear projection approach for 2D features
            B_feat, feat_dim, _ = encoder_features.shape
            encoder_features = encoder_features.view(B_feat, feat_dim)  # [B, feat_dim]
            
            # Create patches from linear projection (MAE-EEG approach)
            num_patches = T // self.patch_size
            feat_patches = encoder_features.unsqueeze(1).expand(B_feat, num_patches, feat_dim)  # [B, num_patches, feat_dim]
            
            # Patch embedding
            feat_patches = self.patch_embed(feat_patches)
            
            # Create mask tokens for masked patches
            mask_patches = mask.view(B, C, -1, self.patch_size).any(dim=1)  # (B, num_patches)
            mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, num_patches, -1)
            
            # Replace masked patches with mask tokens (MAE approach)
            decoder_input = torch.where(mask_patches.unsqueeze(-1), mask_tokens, feat_patches)
            
            # Transformer decoding
            decoded = self.transformer(decoder_input, feat_patches)
            
            # Project to output
            output = self.output_proj(decoded)  # (B, num_patches, patch_size * output_channels)
            output = output.view(B, self.output_channels, T)
        else:
            # Standard 3D features: [B, C, T]
            num_patches = T // self.patch_size
            
            # Reshape encoder features to patches (MAE-EEG approach)
            feat = encoder_features  # (B, feat_dim, L_feat)

            # Adaptive pooling to exactly num_patches tokens (works for L_feat smaller or larger)
            # result: (B, feat_dim, num_patches)
            feat_tokens = F.adaptive_avg_pool1d(feat, output_size=num_patches)

            # Permute to (B, num_patches, feat_dim)
            feat_patches = feat_tokens.permute(0, 2, 1).contiguous()
            #print("FEATURES SHAPE:", feat.shape)
            #print("NUM_PATCHES:", num_patches, "PATCH_SIZE:", self.patch_size)
            #print("FEAT_TOKENS:", feat_patches.shape)
            
            # Patch embedding
            feat_patches = self.patch_embed(feat_patches)
            
            # Create mask tokens for masked patches
            mask_patches = mask.view(B, C, num_patches, self.patch_size).any(dim=1).any(dim=-1)  # (B, num_patches)
            mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, num_patches, -1)

            assert feat_patches.ndim == 3  # (B, num_patches, feat_dim)
            assert mask_patches.shape == (B, feat_patches.shape[1])
            assert mask_tokens.shape == feat_patches.shape
            
            # Replace masked patches with mask tokens (MAE approach)
            decoder_input = torch.where(mask_patches.unsqueeze(-1), mask_tokens, feat_patches)
            
            # Transformer decoding
            decoded = self.transformer(decoder_input, feat_patches)
            
            # Project to output
            output = self.output_proj(decoded)  # (B, num_patches, patch_size * output_channels)
            output = output.view(B, self.output_channels, T)
            #print("1. DEC_OUTPUT:", output.shape)
        
        # Ensure exact output length using adaptive pooling
        # if output.shape[-1] != self.output_length:
        #     output = F.adaptive_avg_pool1d(output, self.output_length)

        output = output.view(B, self.output_channels, -1)

        # If length larger/smaller than T, resample with adaptive avg pool to exactly T
        if output.shape[-1] != self.output_length:
            output = F.adaptive_avg_pool1d(output, self.output_length)

        # --- smoothing: simple grouped conv with small kernel to remove sharp spikes ---
        # create a small smoothing kernel (e.g., triangular / gaussian) on CPU and register buffer
        kernel_size = 5
        pad = kernel_size // 2

        # Register CPU buffer once (float32) if not present
        if not hasattr(self, "_smooth_kernel"):
            k = torch.linspace(1.0, 0.0, steps=kernel_size)
            k = (k + k.flip(0))[:kernel_size]
            k = k / k.sum()
            # store as buffer in float32 on CPU initially
            self.register_buffer("_smooth_kernel", k.view(1, 1, -1).to(torch.float32))

        # Ensure kernel matches output's device & dtype (move/convert only when needed)
        kernel = self._smooth_kernel
        if kernel.device != output.device or kernel.dtype != output.dtype:
            # move/convert and also store back to avoid repeat conversions
            kernel = kernel.to(device=output.device, dtype=output.dtype)
            # replace buffer with converted version so future forwards skip conversion
            # keep name consistent so .state_dict() includes it (good for checkpointing)
            self._smooth_kernel = kernel

        # prepare weight for grouped conv: shape (out_channels, 1, k)
        weight = kernel.expand(self.output_channels, -1, -1)

        # pad & apply grouped conv
        out_padded = F.pad(output, (pad, pad), mode='reflect')
        smoothed = F.conv1d(out_padded, weight, groups=self.output_channels)

        output = smoothed
        #print("2. DEC_OUTPUT:", output.shape)
        
        return output


class MAEPretrainer(pl.LightningModule):
    """MAE-style pretraining wrapper for Braindecode models."""
    
    def __init__(self, plots_dir: str, backbone: nn.Module, patch_size: int = 50, mask_ratio: float = 0.5, 
                 mask_seed: int = None, decoder: str = 'conv', learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-5, num_epochs: int = 100):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_seed = mask_seed
        self.decoder_type = decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.plots_dir = plots_dir
        
        # Initialize masking
        self.masking = PatchMasking(self.patch_size, self.mask_ratio, self.mask_seed)
        
        # Will be initialized after probing
        self.decoder = None
        self.encoder_feat_dim = None
        self.output_channels = None
        self.output_length = None
        
        # Loss function with reduction
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # Flag to track if decoder is initialized
        self._decoder_initialized = False
        
    def _probe_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Probe backbone to get feature dimensions."""
        # Check if backbone is actually a model
        if not hasattr(self.backbone, 'modules'):
            print(f"Warning: Backbone is not a PyTorch model (type: {type(self.backbone)})")
            # Return dummy features with correct shape
            return torch.randn(x.shape[0], 128, x.shape[2] // 4, device=x.device)
        
        with torch.no_grad():
            # Try different methods to get features
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
            else:
                # Use forward hook to capture intermediate features
                features = self._get_features_with_hook(x)
                
        return features
    
    def _get_features_with_hook(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features by removing final classification layers for specific models."""
        # Get model name for specific handling
        model_name = self.backbone.__class__.__name__
        
        if model_name == 'BIOT':
            # BIOT: Remove _ClassificationHead (final_layer)
            original_final_layer = self.backbone.final_layer
            self.backbone.final_layer = nn.Identity()
            features = self.backbone(x)
            self.backbone.final_layer = original_final_layer

        elif model_name == 'BIOT_PRE':
            # BIOT: Remove _ClassificationHead (final_layer)
            original_final_layer = self.backbone.final_layer
            self.backbone.final_layer = nn.Identity()
            features = self.backbone(x)
            self.backbone.final_layer = original_final_layer
            
        elif model_name == 'EEGNet':
            # EEGNet: Remove final_layer (Sequential with conv_classifier)
            original_final_layer = self.backbone.final_layer
            self.backbone.final_layer = nn.Identity()
            features = self.backbone(x)
            self.backbone.final_layer = original_final_layer
            
        elif model_name == 'EEGNeX':
            # EEGNeX: Remove final_layer (ParametrizedLinearWithConstraint)
            print(self.backbone)
            breakpoint()
            original_final_layer = self.backbone.final_layer
            self.backbone.final_layer = nn.Identity()
            features = self.backbone(x)
            self.backbone.final_layer = original_final_layer
            
        elif model_name == 'SPARCNet':
            # SPARCNet: Remove final linear layer
            if hasattr(self.backbone, 'classifier'):
                original_classifier = self.backbone.classifier
                self.backbone.classifier = nn.Identity()
                features = self.backbone(x)
                self.backbone.classifier = original_classifier
            else:
                features = self.backbone(x)
                
        elif model_name == 'Deep4Net':
            # Deep4Net: Remove final linear layer
            if hasattr(self.backbone, 'classifier'):
                original_classifier = self.backbone.classifier
                self.backbone.classifier = nn.Identity()
                features = self.backbone(x)
                self.backbone.classifier = original_classifier
            else:
                features = self.backbone(x)
                
        elif model_name == 'ATCNet':
            # ATCNet: Remove final linear layer
            if hasattr(self.backbone, 'classifier'):
                original_classifier = self.backbone.classifier
                self.backbone.classifier = nn.Identity()
                features = self.backbone(x)
                self.backbone.classifier = original_classifier
            else:
                features = self.backbone(x)
        else:
            # Fallback: Use hooks on conv layers
            features = None
            def hook_fn(module, input, output):
                nonlocal features
                if isinstance(output, tuple):
                    features = output[0]
                else:
                    features = output
            hook_handles = []
            for module in self.backbone.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    hook_handles.append(module.register_forward_hook(hook_fn))
            _ = self.backbone(x)
            for handle in hook_handles:
                handle.remove()
            
            # Fallback to full forward pass output if hooks didn't capture features
            if features is None:
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
        
        return features
    
    def _initialize_decoder(self, x: torch.Tensor):
        """Initialize decoder based on backbone output."""
        if self.decoder is not None:
            return
            
        # Probe backbone
        features = self._probe_backbone(x)
        print("DEBUG: backbone features.shape =", features.shape)
        
        # Handle different feature shapes
        if len(features.shape) == 4:  # [B, C, H, W] -> [B, C, H*W]
            features = features.view(features.shape[0], features.shape[1], -1)
        elif len(features.shape) == 3:  # [B, C, L] - already correct
            pass
        elif len(features.shape) == 2:  # [B, feat_dim] -> [B, feat_dim, 1]
            features = features.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        # Extract dimensions
        self.encoder_feat_dim = features.shape[1]
        self.output_channels = x.shape[1]
        self.output_length = x.shape[2]
        
        
        # Initialize decoder based on specified type
        if self.decoder_type == 'conv':
            self.decoder = ConvDecoder(
                self.encoder_feat_dim, 
                self.output_channels, 
                self.output_length
            )
        elif self.decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                self.encoder_feat_dim,
                self.output_channels,
                self.output_length,
                self.patch_size
            )
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
            
        self.decoder = self.decoder.to(x.device)
        self._decoder_initialized = True
        
    def forward(self, x: torch.Tensor, training: bool = True):
        """Forward pass with masking and reconstruction."""
        # Store original input for comparison
        original_x = x.clone()
        B,C,T = original_x.shape
        
        # Initialize decoder if needed
        if self.decoder is None:
            self._initialize_decoder(x)
            
        # Apply masking
        masked_x, mask = self.masking(x, training)
        
        # Get encoder features
        # if hasattr(self.backbone, 'forward_features'):
        #     features = self.backbone.forward_features(masked_x)
        # else:
        features = self._get_features_with_hook(masked_x)
        
        # Handle different feature shapes for MAE-EEG reconstruction
        if len(features.shape) == 4:  # [B, C, H, W] -> [B, C, H*W]
            features = features.view(features.shape[0], features.shape[1], -1)
        elif len(features.shape) == 2:  # [B, feat_dim] -> [B, feat_dim, 1]
            # For 2D features (EEGNeX, BIOT), create temporal dimension
            # This follows MAE-EEG approach where global features are used for reconstruction
            features = features.unsqueeze(-1)  # [B, feat_dim, 1]
        elif len(features.shape) == 3:  # [B, C, L] - already correct
            pass
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        #reconstructed = features[:,:,:T]

        #Decode to reconstruction using specified decoder type
        if self.decoder_type == 'conv':
            reconstructed = self.decoder(features)
        else:  # transformer
            reconstructed = self.decoder(features, mask)
            
        return original_x, reconstructed, mask
    
    def training_step(self, batch, batch_idx):
        # Handle different batch formats - take first element as input
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Take first element (EEG data)
        else:
            x = batch
        
        original, reconstructed, mask = self.forward(x, training=True)
        
        # Simple, foolproof loss computation
        mask_count = mask.sum()
        if mask_count == 0:
            # If no masking, use whole signal
            loss = F.mse_loss(reconstructed, original, reduction='mean')
        else:
            # Use only masked regions (MAE approach)
            loss = F.mse_loss(reconstructed[mask], original[mask], reduction='mean')
        
        # Debug prints for first batch only
        if batch_idx == 0:
            print(f"TRAIN - Original: [{original.min():.4f}, {original.max():.4f}], mean={original.mean():.4f}")
            print(f"TRAIN - Reconstructed: [{reconstructed.min():.4f}, {reconstructed.max():.4f}], mean={reconstructed.mean():.4f}")
            print(f"TRAIN - Loss: {loss:.6f}, Mask ratio: {mask.float().mean():.4f}")
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mask_ratio', mask.float().mean(), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Handle different batch formats - take first element as input
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Take first element (EEG data)
        else:
            x = batch
        
        # Store batch for plotting (only first batch of each epoch)
        if batch_idx == 0:
            self._current_val_batch = x

        # store for plotting later
        self._current_val_batch_raw = x.detach().clone().cpu()

        # now call forward using a device copy
        x_device = x.to(self.device)
        original, reconstructed, mask = self.forward(x_device, training=False)
        
        # Simple, foolproof loss computation (same as training)
        mask_count = mask.sum()
        if mask_count == 0:
            # If no masking, use whole signal
            loss = F.mse_loss(reconstructed, original, reduction='mean')
        else:
            # Use only masked regions (MAE approach)
            loss = F.mse_loss(reconstructed[mask], original[mask], reduction='mean')
        
        # Debug prints for validation
        print(f"VAL - Original: [{original.min():.4f}, {original.max():.4f}], mean={original.mean():.4f}")
        print(f"VAL - Reconstructed: [{reconstructed.min():.4f}, {reconstructed.max():.4f}], mean={reconstructed.mean():.4f}")
        print(f"VAL - Loss: {loss:.6f}, Mask ratio: {mask.float().mean():.4f}")
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_validation_epoch_end(self):
        """Plot reconstructions during validation."""
        # Only plot during actual training, not during sanity check
        if hasattr(self, 'current_epoch') and self.current_epoch % 5 == 0:
            self._plot_reconstructions()
    
    def _plot_reconstructions(self):
        """Plot original vs reconstructed signals using validation batch."""
        from pathlib import Path
        from ..plot import plot_mae_reconstruction, plot_mae_reconstructions
        
        # We're already in eval mode during validation
        with torch.no_grad():
            # Get the current validation batch (passed as parameter)
            # We need to access the validation data from the current batch
            # This will be called during validation_step, so we can store the batch
            if hasattr(self, '_current_val_batch'):
                x = self._current_val_batch
                
                # Take first 3 samples (or all if less than 3)
                #x = x[:min(3, x.shape[0])]
                #x_raw = x[:min(3, x.shape[0])].detach().cpu()
                
                #original, reconstructed, mask = self.forward(x, training=False)

                #x_raw = self._current_val_batch_raw  # already cpu, detached
                if not hasattr(self, "_current_val_batch_raw"):
                    print("No saved raw batch.")
                    return
                x_raw = self._current_val_batch_raw  # already cpu, detached
                # move a small selection to device
                x_device = x_raw[:min(3, x_raw.shape[0])].to(self.device)
                with torch.no_grad():
                    original, reconstructed, mask = self.forward(x_device, training=False)
                    original_sample, reconstructed_sample, mask_sample = self.forward(x_raw.to(self.device), training=False)

                # Loss for 3 samples
                loss_compare = torch.linalg.norm((original_sample-reconstructed_sample), axis=(1,2))
                sample = torch.argmin(loss_compare)
                print(f"sample: {sample} | loss: {loss_compare[sample]}")
                
                # Create plots directory
                self.plots_dir.mkdir(exist_ok=True)
                
                # Use the new analysis plotting function
                plot_mae_reconstruction(x_raw, reconstructed.cpu(), mask.cpu(), self.current_epoch, self.plots_dir)
                plot_mae_reconstructions(x_raw, reconstructed_sample.cpu(), mask_sample.cpu(), self.current_epoch, self.plots_dir, sample_idx=sample)
            else:
                print("Warning: No validation batch available for plotting")
    
