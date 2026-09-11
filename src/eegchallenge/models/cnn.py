# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class eConvNet(nn.Module):
    def __init__(self, n_chans=129, n_outputs=1, n_times=200):
        super(eConvNet, self).__init__()
        #CNN Block 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(129, 128, kernel_size=13, stride=1),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            #SEBlock1d(128)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=13, stride=1),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            #SEBlock1d(256)
            )
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=9, stride=1),
            #nn.BatchNorm1d(512),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=7, stride=1),
            #nn.BatchNorm1d(512),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # Dropout
        self.bilstm = nn.LSTM(input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            #dropout=0.5
        )
        self.attn_conv = nn.Sequential(
            nn.Conv1d(512,1,kernel_size=1))      
        self.drop_out = nn.Dropout(p=0.5)
        # Output FC Layer
        self.fc1 = nn.Linear(512, 1)
    def forward(self,x):
        out = self.layer1(x)
        #out = self.drop_out(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.drop_out(out)
        out = self.layer4(out)
        #out = self.layer5(out) 
        out = self.layer6(out)
        
        #out = self.drop_out(out)
        #x, _ = self.bilstm(out.permute(0,2,1))
        #out = x.permute(0,2,1); 

        #attn = self.attn_conv(out)
        #attn = torch.softmax(attn, dim=2)
        #out = torch.sum(out * attn, dim=2)
        
        # Temporal Averaging
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, n_chans=129, n_outputs=1, n_times=200):
        super(ConvNet, self).__init__()
        #CNN Block 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(129, 128, kernel_size=13, stride=1),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            #SEBlock1d(128)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=13, stride=1),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            #SEBlock1d(256)
            )
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=9, stride=1),
            #nn.BatchNorm1d(512),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=7, stride=1),
            #nn.BatchNorm1d(512),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # Dropout
        self.bilstm = nn.LSTM(input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            #dropout=0.5
        )
        self.attn_conv = nn.Sequential(
            nn.Conv1d(512,1,kernel_size=1))      
        self.drop_out = nn.Dropout(p=0.5)
        # Output FC Layer
        self.fc1 = nn.Linear(512, 1)
    def forward(self,x):
        out = self.layer1(x)
        #out = self.drop_out(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.drop_out(out)
        out = self.layer4(out)
        #out = self.layer5(out) 
        out = self.layer6(out)
        
        out = self.drop_out(out)
        x, _ = self.bilstm(out.permute(0,2,1))
        out = x.permute(0,2,1); 

        #attn = self.attn_conv(out)
        #attn = torch.softmax(attn, dim=2)
        #out = torch.sum(out * attn, dim=2)
        
        # Temporal Averaging
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out

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

# class ResBlock1d(nn.Module):
#     def __init__(self, channels, kernel_size):
#         super().__init__()
#         pad = kernel_size//2
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
#         self.bn1   = nn.BatchNorm1d(channels)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
#         self.bn2   = nn.BatchNorm1d(channels)
#         self.elu   = nn.ELU()
#     def forward(self, x):
#         identity = x
#         out = self.elu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         return self.elu(out + identity)

class ParallelDeepBlock(nn.Module):
    """
    A 4-layer deep parallel block.  
    - in_ch → out_ch total, split evenly across B branches.
    - Each branch: conv→BN→ReLU→conv→BN→ReLU→pool→conv→BN→ReLU→conv→BN→ReLU→pool
    """
    def __init__(self, in_ch, out_ch, kernel_sizes, pool_every=2):
        super().__init__()
        B = len(kernel_sizes)
        assert out_ch % B == 0, "out_ch must divide evenly by number of branches"
        branch_ch = out_ch // B
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = []
            # two 1D convs then pool, then two convs then pool
            for i, ksize in enumerate([k, k]):
                layers += [
                    nn.Conv1d(in_ch if i==0 else (branch_ch // 2), (branch_ch // 2),
                              kernel_size=ksize, padding=ksize//2, bias=False),
                    nn.BatchNorm1d(branch_ch//2),
                    nn.ELU()
                ]
                if (i+1) % pool_every == 0:
                    layers.append(nn.MaxPool1d(kernel_size=4, stride=2))
            # repeat depth-2 more convs
            for i, ksize in enumerate([k//2, k//2]):
                layers += [
                    nn.Conv1d((branch_ch//2) if i==0 else branch_ch, branch_ch,
                              kernel_size=ksize, padding=ksize//2, bias=False),
                    nn.BatchNorm1d(branch_ch),
                    nn.ELU()
                ]
                if ((i+1)+2) % pool_every == 0:
                    layers.append(nn.MaxPool1d(kernel_size=4, stride=2))
            self.branches.append(nn.Sequential(*layers))

        # fuse concatenated branches
        self.fuse    = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(out_ch)
        self.act     = nn.ELU()

    def forward(self, x):
        # x: [B, in_ch, T]
        outs = [b(x) for b in self.branches]       # each [B, branch_ch, T']
        Ts = [o.size(2) for o in outs]; T_min = min(Ts)
        outs = [o[:, :, :T_min] for o in outs]
        x_cat = torch.cat(outs, dim=1)             # [B, out_ch, T']
        return self.act(self.bn_fuse(self.fuse(x_cat)))

class AsdNet(nn.Module):
    def __init__(self,
                 n_chans=129,
                 sample_rate=100,
                 kernels=(7, 13, 25, 51),
                 freq_range=(4,60),
                 depth_ch=512,
                 use_se=True,
                 use_lstm=True,
                 use_transformer=False,
                 infer_kernels=False,
                 use_depthwise=False,
                 use_channel_drop=True,
                 lstm_hidden=256,
                 trans_layers=1,
                 trans_heads=4,
                 use_temporal_avg=True,
                 dropout=0.5,
                 n_outputs=1,
                 n_times=200):
        super().__init__()

        self.sample_rate = sample_rate
        self.channel_drop = nn.Sequential(
            nn.Identity() if not use_channel_drop else nn.Dropout1d(p=0.5))
        # spatial filter: mixes channels per timepoint
        self.spatial = nn.Conv1d(n_chans, n_chans, kernel_size=1, bias=False)

        if use_depthwise:
            # depthwise: groups=in_channels
            self.depthwise = nn.Conv1d(n_chans, n_chans, kernel_size=3, padding=1,
                                       groups=n_chans, bias=False)
        else:
            self.depthwise = nn.Identity()

        if infer_kernels:
            f_low, f_high = freq_range
            # convert Hz→samples for half‐width kernels
            k_lo = max(3, int(sample_rate / f_high))   # smallest kernel
            k_hi = max(k_lo+2, int(sample_rate / f_low))  # largest
            # build a geometric progression of odd kernel sizes
            kernels = []
            for alpha in [0.25, 0.5, 0.75, 1.0]:
                k = int(k_lo + alpha*(k_hi-k_lo))
                if k % 2 == 0: k += 1
                kernels.append(k)
            kernels = sorted(set(kernels))

        # deep parallel block capturing multiple receptive fields
        self.parallel = ParallelDeepBlock(in_ch=n_chans,
                                          out_ch=depth_ch,
                                          kernel_sizes=kernels)

        # optional SE block
        self.se = SEBlock1d(depth_ch) if use_se else nn.Identity()

        # Use LSTM or Transformer
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_size=depth_ch,
                                hidden_size=lstm_hidden,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
            post_feat = 2 * lstm_hidden
        else:
            post_feat = depth_ch

        self.use_trans = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=depth_ch,
                                                       nhead=trans_heads,
                                                       dim_feedforward=depth_ch,
                                                       dropout=dropout,
                                                       activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=trans_layers)
            post_feat = depth_ch

        self.attn = nn.Conv1d(post_feat, 1, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.use_temporal_avg = use_temporal_avg

        #final classifier
        self.fc = nn.Linear(post_feat, 1)

    def forward(self, x):
        # x: [B, chans, T]
        x = self.channel_drop(x)   # random **channel** dropout
        x = self.spatial(x)        # mixing across channels
        x = self.depthwise(x)      # local per-channel filtering

        x = self.parallel(x)                      # deep parallel features
        x = self.se(x)                            # optional SE

        # flatten for LSTM/Transformer: bring time to second dim
        if self.use_lstm:
            # BiLSTM expects [B, T', C]
            #x = self.drop(x)
            x = x.transpose(1,2)                  # [B, T', C]
            x, _ = self.lstm(x)                   # [B, T', 2*h]
            x = x.transpose(1,2)                  # back to [B, C', T']
        elif self.use_trans:
            # Transformer wants [T', B, C]
            x = x.permute(2,0,1)
            x = self.transformer(x)               # [T', B, C]
            x = x.permute(1,2,0)                  # [B, C, T']

        # self-attention pooling
        if self.use_temporal_avg:
            x = x.mean(axis=2)
        else:
            attn_w = torch.softmax(self.attn(x), dim=2)  # [B,1,T']
            x = torch.sum(x * attn_w, dim=2)             # [B, C']
        x = self.drop(x)

        logits = self.fc(x)                          # [B,2]
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

class stConvNet(nn.Module):
    """
    A CNN that first learns temporal patterns and then spatial (channel)
    patterns, inspired by separable convolutions used in models like EEGNet.
    """
    def __init__(self, n_chans=129, n_times=200,  n_outputs=1, temporal_filters=16):
        super(stConvNet, self).__init__()
        
        self.n_chans = n_chans
        self.temporal_filters = temporal_filters

        # 1. Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, temporal_filters, kernel_size=(1, 25), 
                                       padding=(0, 12)) # (k-1)/2
        self.bn_temporal = nn.BatchNorm2d(temporal_filters)
        
        # 2. Spatial (Depthwise) Convolution
        self.spatial_conv = nn.Conv2d(temporal_filters, temporal_filters, 
                                      kernel_size=(n_chans, 1), 
                                      groups=temporal_filters)
        self.bn_spatial = nn.BatchNorm2d(temporal_filters)
        
        # Calculate flattened feature size
        # After temporal_conv: (B, 16, C, T)
        # After spatial_conv: (B, 16, 1, T)
        # After avg_pool: (B, 16, 1, T/4)
        # We flatten 16 * (T/4)
        pooled_time = n_times // 4
        flattened_size = temporal_filters * pooled_time
        
        # Regression Head
        self.pooling = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, 1) # Output is a single value

    def forward(self, x):
        # Input x shape: (B, C, T) -> (B, 129, 200)
        
        # Reshape for Conv2d: (B, 1, C, T)
        # (Batch, In_Channels, Height, Width)
        x = x.unsqueeze(1) 
        
        # 1. Temporal Convolution
        # (B, 1, 129, 200) -> (B, 16, 129, 200)
        x = self.temporal_conv(x)
        x = F.elu(self.bn_temporal(x))
        
        # 2. Spatial Convolution (Depthwise)
        # (B, 16, 129, 200) -> (B, 16, 1, 200)
        x = self.spatial_conv(x)
        x = F.elu(self.bn_spatial(x))
        
        # 3. Pooling and Flattening
        # (B, 16, 1, 200) -> (B, 16, 1, 50)
        x = self.pooling(x)
        
        # (B, 16, 1, 50) -> (B, 16 * 50) = (B, 800)
        x = torch.flatten(x, 1)
        
        # 4. Regression Head
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.fc2(x) # Final prediction
        
        return x