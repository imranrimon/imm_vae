
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.simple_ae import UNetDownBlock, UNetUpBlock

class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, z_dim=128, input_res=64):
        super().__init__()
        self.input_res = input_res
        
        # Bottom-up Encoder
        # Level 1 (Fine): 64 -> 32 -> 16
        # Or 32 -> 16 -> 8 (for CIFAR)
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # Block 1: InputRes -> InputRes/2
        self.down1 = UNetDownBlock(base_channels, base_channels, down=True) 
        
        # Block 2: InputRes/2 -> InputRes/4
        self.down2 = UNetDownBlock(base_channels, base_channels*2, down=True)
        
        # Level 2 (Coarse): InputRes/4 -> InputRes/8
        self.down3 = UNetDownBlock(base_channels*2, base_channels*4, down=True)
        
        # Block 4: InputRes/8 -> InputRes/16 (Global Feature Map)
        self.down4 = UNetDownBlock(base_channels*4, base_channels*4, down=True)
        
        # If Input=64: 64->32->16->8->4. Feature map 4x4.
        # If Input=32: 32->16->8->4->2. Feature map 2x2.
        
        final_res = input_res // 16
        flat_dim = base_channels*4 * final_res * final_res
        
        self.to_global_mean = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, z_dim)
        )
        self.to_global_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, z_dim)
        )
        
        # Fine stats from Block 2 output (16x16 or 8x8)
        # channels = base_channels*2
        self.to_fine_stats = nn.Sequential(
             nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
             nn.Conv2d(base_channels*2, base_channels*2 * 2, 1) # Mean/Logvar
        )

    def forward(self, x):
        h = self.conv_in(x)
        h1, feat1 = self.down1(h) # feat1 is pre-downsample? UNetDown return down(h), h
        # Wait, UNetDownBlock returns (down(h), h) where h is output of net (before down), down(h) is next input.
        # feat1 = h (same res as input to block). We want 'down1' output.
        # No, 'feat1' (h) is used for skips usually.
        # Here we want the downsampled output to continue.
        # h1 is downsampled.
        
        h2, feat2 = self.down2(h1) # h2 is Res/4. feat2 is Res/2 [use for fine stats?]
        # Fine stats usually at higher res. Let's use feat2 (Res/2) or h2 (Res/4)?
        # Let's use h2 (Res/4).
        
        h3, feat3 = self.down3(h2) # h3 is Res/8
        h4, feat4 = self.down4(h3) # h4 is Res/16
        
        # Coarse
        mu_global = self.to_global_mean(h4)
        logvar_global = self.to_global_logvar(h4)
        
        # Fine (Spatial) - Attached at h2 (Res/4)
        # For CelebA: 16x16. For CIFAR: 8x8.
        stats_fine = self.to_fine_stats(h2)
        mu_fine, logvar_fine = stats_fine.chunk(2, dim=1)
        
        return mu_global, logvar_global, mu_fine, logvar_fine, h2, h4


class HierarchicalDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, z_dim=128, input_res=64):
        super().__init__()
        
        final_res = input_res // 16
        self.spatial_dim = final_res
        self.base_channels = base_channels
        
        self.from_global = nn.Sequential(
            nn.Linear(z_dim, base_channels*4 * final_res * final_res),
            nn.SiLU()
        )
        
        # Up from Res/16 -> Res/8
        self.up4 = UNetUpBlock(base_channels*4, base_channels*4, up=True)
        
        # Up from Res/8 -> Res/4
        self.up3 = UNetUpBlock(base_channels*4, base_channels*2, up=True)
        
        # At Res/4, we combine with Z_fine.
        # Z_fine channels = base_channels*2 * 2? No, z_fine dim matches feature dim usually?
        # In Encoder 'to_fine_stats' output 2*channels. So z_fine has 'base_channels*2' channels.
        
        # Prior p(z_fine | z_global features)
        self.prior_fine_stats = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.Conv2d(base_channels*2, base_channels*2 * 2, 1)
        )
        
        # After concat z_fine (channels*2) + feature (channels*2) = channels*4
        # We need to reduce or handle in next block.
        # Next block 'up2' goes Res/4 -> Res/2. 
        # Input channels will be base_channels*4.
        self.up2 = UNetUpBlock(base_channels*4, base_channels, up=True) 
        
        # Up from Res/2 -> Res
        self.up1 = UNetUpBlock(base_channels, base_channels, up=True)
        
        self.out_conv = nn.Sequential(
             nn.Conv2d(base_channels, out_channels, 3, 1, 1),
             nn.Tanh()
        )

    def forward(self, z_global, z_fine=None):
        # 1. Process Global
        h = self.from_global(z_global)
        h = h.view(-1, self.base_channels*4, self.spatial_dim, self.spatial_dim)
        
        h = self.up4(h) # Res/8 [B, 256, H/8, W/8]
        h_coarse = self.up3(h) # Res/4 [B, 128, H/4, W/4]
        
        # 2. Process Fine
        # Prior p(z_fine | z_global)
        stats_prior = self.prior_fine_stats(h_coarse)
        mu_prior, logvar_prior = stats_prior.chunk(2, dim=1)
        
        if z_fine is None:
            std = torch.exp(0.5 * logvar_prior)
            z_fine = mu_prior + torch.randn_like(mu_prior) * std
            
        # Combine
        combined = torch.cat([h_coarse, z_fine], dim=1) # 128+128=256
        
        # 3. Final Decode
        h = self.up2(combined) # Res/2
        h = self.up1(h) # Res
        out = self.out_conv(h)
        
        return out, mu_prior, logvar_prior

class NVAE(nn.Module):
    def __init__(self, z_dim=128, input_res=64):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = HierarchicalEncoder(z_dim=z_dim, input_res=input_res)
        self.decoder = HierarchicalDecoder(z_dim=z_dim, input_res=input_res)
    
    def forward(self, x):
        # 1. Encode
        mu_g, log_g, mu_f, log_f, _, _ = self.encoder(x)
        
        # 2. Reparameterize Global
        std_g = torch.exp(0.5 * log_g)
        z_g = mu_g + torch.randn_like(mu_g) * std_g
        
        # 3. Reparameterize Fine (Posterior)
        std_f = torch.exp(0.5 * log_f)
        z_f = mu_f + torch.randn_like(mu_f) * std_f
        
        # 4. Decode
        x_rec, mu_prior_f, logvar_prior_f = self.decoder(z_g, z_f)
        
        return {
            'x_rec': x_rec,
            'mu_g': mu_g, 'log_g': log_g, 'z_g': z_g,
            'mu_f': mu_f, 'log_f': log_f, 'z_f': z_f,
            'mu_prior_f': mu_prior_f, 'log_prior_f': logvar_prior_f
        }
