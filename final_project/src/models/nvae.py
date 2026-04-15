
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.down = down
        if down:
            self.pool = nn.Conv2d(out_channels, out_channels, 4, 2, 1) # Downsample
        
    def forward(self, x):
        h = self.conv(x)
        if self.down:
            return self.pool(h), h # Return downsampled and featured
        return h

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()
        self.up = up
        if up:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        if self.up:
            h = self.up_conv(x)
        else:
            h = self.conv(x)
        return self.norm(h)

class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, z_dim=128, input_res=64):
        super().__init__()
        self.input_res = input_res
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # 32 -> 16
        self.down1 = UNetDownBlock(base_channels, base_channels, down=True) 
        # 16 -> 8
        self.down2 = UNetDownBlock(base_channels, base_channels*2, down=True)
        # 8 -> 4
        self.down3 = UNetDownBlock(base_channels*2, base_channels*4, down=True)
        # 4 -> 2
        self.down4 = UNetDownBlock(base_channels*4, base_channels*4, down=True)
        
        final_res = 2
        flat_dim = base_channels*4 * final_res * final_res
        
        self.to_global_mean = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, z_dim)
        )
        self.to_global_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, z_dim)
        )
        
        # Fine stats attached at Res/4 (8x8) -> Use h2 output (Res/4 if input 32)
        # Wait, if input 32: 32 -> 16 (h1) -> 8 (h2) -> 4 (h3) -> 2 (h4)
        # Let's use h2 (8x8) for fine stats
        self.to_fine_stats = nn.Sequential(
             nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
             nn.Conv2d(base_channels*2, base_channels*2 * 2, 1) 
        )

    def forward(self, x):
        h = self.conv_in(x)
        h1, _ = self.down1(h) # 16
        h2, _ = self.down2(h1) # 8
        h3, _ = self.down3(h2) # 4
        h4, _ = self.down4(h3) # 2
        
        # Coarse
        mu_global = self.to_global_mean(h4)
        logvar_global = self.to_global_logvar(h4)
        logvar_global = torch.clamp(logvar_global, -10., 10.)
        
        # Fine (Spatial)
        stats_fine = self.to_fine_stats(h2)
        mu_fine, logvar_fine = stats_fine.chunk(2, dim=1)
        logvar_fine = torch.clamp(logvar_fine, -10., 10.)
        
        return mu_global, logvar_global, mu_fine, logvar_fine, h2, h4


class HierarchicalDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, z_dim=128, input_res=64):
        super().__init__()
        
        final_res = 2
        self.spatial_dim = final_res
        self.base_channels = base_channels
        
        self.from_global = nn.Sequential(
            nn.Linear(z_dim, base_channels*4 * final_res * final_res),
            nn.SiLU()
        )
        
        # 2 -> 4
        self.up4 = UNetUpBlock(base_channels*4, base_channels*4, up=True)
        # 4 -> 8
        self.up3 = UNetUpBlock(base_channels*4, base_channels*2, up=True)
        
        # Prior p(z_fine | z_global features) at 8x8
        self.prior_fine_stats = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.Conv2d(base_channels*2, base_channels*2 * 2, 1)
        )
        
        # 8 -> 16 (Combined features)
        self.up2 = UNetUpBlock(base_channels*4, base_channels, up=True) 
        # 16 -> 32
        self.up1 = UNetUpBlock(base_channels, base_channels, up=True)
        
        self.out_conv = nn.Sequential(
             nn.Conv2d(base_channels, out_channels, 3, 1, 1),
             nn.Tanh()
        )

    def forward(self, z_global, z_fine=None):
        # 1. Process Global
        h = self.from_global(z_global)
        h = h.view(-1, self.base_channels*4, self.spatial_dim, self.spatial_dim) # 2x2
        
        h = self.up4(h) # 4x4
        h_coarse = self.up3(h) # 8x8
        
        # 2. Process Fine
        stats_prior = self.prior_fine_stats(h_coarse)
        mu_prior, logvar_prior = stats_prior.chunk(2, dim=1)
        logvar_prior = torch.clamp(logvar_prior, -10., 10.)
        
        if z_fine is None:
            std = torch.exp(0.5 * logvar_prior)
            z_fine = mu_prior + torch.randn_like(mu_prior) * std
            
        # Combine
        combined = torch.cat([h_coarse, z_fine], dim=1) 
        
        # 3. Final Decode
        h = self.up2(combined) # 16x16
        h = self.up1(h) # 32x32
        out = self.out_conv(h)
        
        return out, mu_prior, logvar_prior

class NVAE(nn.Module):
    def __init__(self, z_dim=128, input_res=32):
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
