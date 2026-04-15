
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(self.hidden_dim, dim, 1), LayerNorm2d(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h*w), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, self.hidden_dim, h, w)
        return self.to_out(out) + x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )
        self.down = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down else nn.Identity()

    def forward(self, x):
        h = self.net(x)
        return self.down(h), h

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1) if up else nn.Identity()
        
        # Fixed channel logic from repro script
        # If up=True, Convtranspose output channels = in_channels // 2
        # So net must take in_channels // 2
        net_in_channels = in_channels // 2 if up else in_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(net_in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x, skip=None):
        if isinstance(self.up, nn.ConvTranspose2d):
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, z_dim=128, in_channels=3):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.down1 = UNetDownBlock(32, 64)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256, down=True)
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, z_dim * 2) # Output mean and logvar for VAE
        )

    def forward(self, x):
        x = self.init_conv(x)
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        _, s3 = self.down3(x)
        stats = self.mid(s3)
        mean, logvar = stats.chunk(2, dim=1)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim=128, out_channels=3, output_res=64):
        super().__init__()
        self.output_res = output_res
        self.map = nn.Sequential(
            nn.Linear(z_dim, 256*8*8),
            nn.Unflatten(1, (256, 8, 8))
        )
        self.up1 = UNetUpBlock(256, 128) # 8->16
        self.up2 = UNetUpBlock(128, 64)  # 16->32
        
        # Flexible upsampling based on resolution
        self.up3 = None
        self.up4 = None
        
        final_dim = 64
        
        if output_res >= 64:
            self.up3 = UNetUpBlock(64, 32)   # 32->64
            final_dim = 32
            
        if output_res >= 128:
            self.up4 = UNetUpBlock(32, 16)   # 64->128
            final_dim = 16
            
        self.out = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(final_dim, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.map(z)
        x = self.up1(x) 
        x = self.up2(x)
        if self.up3: x = self.up3(x)
        if self.up4: x = self.up4(x)
        return self.out(x)

class Model(nn.Module):
    def __init__(self, z_dim=128, in_channels=3, input_res=64):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, in_channels=in_channels)
        self.decoder = Decoder(z_dim, out_channels=in_channels, output_res=input_res)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_rec = self.decoder(z)
        return z, x_rec, mean, logvar
