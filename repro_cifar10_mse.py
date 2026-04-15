import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import sqrtm
import os

# ────────────────────────────────────────────────────────────────
# 1) Schedulers for the flow matching / diffusion
# ────────────────────────────────────────────────────────────────
class IMMSchedule:
    def __init__(self):
        pass
    def alpha(self, t):
        raise NotImplementedError
    def sigma(self, t):
        raise NotImplementedError
    def lamb(self, t):
        # log SNR = 2 * (log(alpha) - log(sigma))
        return 2.0 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))
    def lamb_diff(self, t):
        # d/dt (log SNR) roughly
        # For simplicity, exact derivative often used or finite diff.
        # User notebooks usually implement analytic forms.
        raise NotImplementedError
    def eta(self, t):
        raise NotImplementedError
    def eta_inv(self, val):
        raise NotImplementedError

class LinearSchedule(IMMSchedule):
    def __init__(self, s0=1.0, s1=0.01):
        super().__init__()
        self.s0 = s0
        self.s1 = s1
    
    def alpha(self, t):
        # alpha_t = 1 - (1-s0)*t ??? No, often just 1 or t
        # Simple Flow Matching: alpha = 1, sigma = 1-t (or similar)
        # Let's assume standard conditional FM: x_t = (1-t)x0 + t x1
        # Here we follow notebook implementations if possible.
        # Notebook output generally implies alpha=1, sigma=t or similar.
        raise NotImplementedError

# Using the definitions found in the notebook
class OTFMSchedule(IMMSchedule):
    def __init__(self):
        super().__init__()
        self.tmin = 1e-5
        self.tmax = 1.0 - 1e-5

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return 1.0 - t

    def lamb(self, t):
        return -2.0 * torch.log(self.sigma(t))

    def lamb_diff(self, t):
        # d/dt (-2 ln(1-t)) = -2 * (-1/(1-t)) = 2/(1-t)
        return 2.0 / (1.0 - t)

    def eta(self, t):
        return -torch.log(1.0 - t)

    def eta_inv(self, val):
        return 1.0 - torch.exp(-val)

class CosineSchedule(IMMSchedule):
    def __init__(self):
        super().__init__()
        self.tmin = 1e-5
        self.tmax = 1.0 - 1e-5

    def alpha(self, t):
        return torch.cos(0.5 * math.pi * t)

    def sigma(self, t):
        return torch.sin(0.5 * math.pi * t)

    def lamb(self, t):
        return 2.0 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def lamb_diff(self, t):
        # d/dt (2 ln cot(pi/2 t))
        # = 2 * (1/cot) * (-csc^2) * pi/2
        # = -pi / (cos * sin)
        # = -pi / (0.5 * sin(pi*t))
        return -math.pi / (self.alpha(t) * self.sigma(t))

    def eta(self, t):
        # -log tan(...)
        return -torch.log(torch.tan(0.5 * math.pi * t))

    def eta_inv(self, val):
        return 2.0 / math.pi * torch.atan(torch.exp(-val))

# ────────────────────────────────────────────────────────────────
# 2) Basic Layers
# ────────────────────────────────────────────────────────────────
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
        self.net = nn.Sequential(
            nn.Conv2d(in_channels if not up else in_channels//2, out_channels, 3, padding=1),
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
    def __init__(self, z_dim=2):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.down1 = UNetDownBlock(32, 64)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256, down=False)
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, z_dim)
        )

    def forward(self, x):
        # x: [B, 3, 28, 28] maybe? CIFAR is 32x32 usually but notebook says 28?
        # The notebook code later calls transforms.ToTensor() on CIFAR10.
        # CIFAR10 is 32x32.
        x = self.init_conv(x)
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        _, s3 = self.down3(x)
        z = self.mid(s3)
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(z_dim, 256*8*8),
            nn.Unflatten(1, (256, 8, 8))
        )
        # CIFAR10: 32x32
        # If z mapped to 8x8:
        # 8 -> 16 -> 32
        self.up1 = UNetUpBlock(256, 128) # 8->16
        self.up2 = UNetUpBlock(128, 64)  # 16->32
        self.out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh() # Images in [-1, 1] often? Or [0,1]
            # Notebook used MSE with x directly.
            # If x in [0,1] (ToTensor), then Tanh is [-1,1].
            # Training often assumes specific scaling.
            # Let's check notebook data loading.
        )

    def forward(self, z):
        x = self.map(z)
        x = self.up1(x) 
        x = self.up2(x)
        return self.out(x)

class Model(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

def compute_mmd(x, y):
    # Simplified MMD
    x_kernel = x @ x.t()
    y_kernel = y @ y.t()
    xy_kernel = x @ y.t()
    rx = (x_kernel.diag().unsqueeze(0).expand_as(x_kernel))
    ry = (y_kernel.diag().unsqueeze(0).expand_as(y_kernel))
    
    dxx = rx.t() + rx - 2.0 * x_kernel
    dyy = ry.t() + ry - 2.0 * y_kernel
    dxy = rx.t() + ry - 2.0 * xy_kernel
    
    XX, YY, XY = (torch.zeros(xx.shape).to(x.device) for xx in [dxx, dyy, dxy])
    
    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)
        
    return torch.mean(XX + YY - 2.0 * XY)

class IMMParameterization(nn.Module):
    def __init__(self, sigma_data: float, schedule: IMMSchedule):
        super().__init__()
        self.sigma_data = sigma_data
        self.schedule = schedule

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:
        return 1000.0 * t

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        a = self.schedule.alpha(t)
        s = self.schedule.sigma(t)
        return 1.0 / (self.sigma_data * torch.sqrt(a*a + s*s))

    def c_skip(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def c_out(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class OtFmEulerFmParameterization(IMMParameterization):
    def __init__(self, sigma_data: float):
        super().__init__(sigma_data, OTFMSchedule())

    def c_skip(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # always 1
        return torch.ones_like(s)

    def c_out(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # -(t - s) * sigma_data
        return -(t - s) * self.sigma_data

class ResBlock(nn.Module):
    """A single residual block: LayerNorm → Linear → SiLU → Linear → skip."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * 2)
        self.fc2  = nn.Linear(dim * 2, dim)
        self.act  = nn.SiLU()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h

class ResidualDenoiser(nn.Module):
    def __init__(self,
                 input_dim:  int,
                 model_dim:  int = 1024,
                 n_blocks:   int = 8,
                 output_dim: int = None):
        super().__init__()
        output_dim = output_dim or (input_dim - 2)  # D = input_dim-2
        # initial projection
        self.project_in  = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU()
        )
        # residual blocks
        self.blocks = nn.ModuleList([ResBlock(model_dim) for _ in range(n_blocks)])
        # final projection
        self.project_out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        # x: [B, D+2]
        h = self.project_in(x)           # [B, model_dim]
        for block in self.blocks:
            h = block(h)                 # [B, model_dim]
        return self.project_out(h)       # [B, D]

class IMM(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 parameterization: IMMParameterization,
                 a: int = 1,
                 b: int = 4,
                 eta_max: float = 160.0,
                 eta_min: float = 1e-8,
                 k: int = 12,
                 M: int = 16,
                 D: int = 2):
        super().__init__()
        self.network = network
        self.parameterization = parameterization
        self.schedule = parameterization.schedule

        self.D = D
        self.a = a
        self.b = b
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.k = k
        self.M = M

    def forward(self, x_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, D], s,t: [B,1]
        skip = self.parameterization.c_skip(s, t) * x_t  # [B, D]

        # Build model input = [ x_t * c_in(t), c_noise(s), c_noise(t) ]
        c_in_t = self.parameterization.c_in(t)       # [B, 1]
        x_scaled = x_t * c_in_t                       # [B, D]
        noise_s = self.parameterization.c_noise(s)    # [B, 1]
        noise_t = self.parameterization.c_noise(t)    # [B, 1]
        inp = torch.cat([x_scaled, noise_s, noise_t], dim=-1)  # [B, D+2]

        out = self.network(inp)                               # [B, D]
        out = self.parameterization.c_out(s, t) * out         # [B, D]
        return skip + out

    def _ddim(self,
              x_source: torch.Tensor,
              x_target: torch.Tensor,
              s: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
        # x_source @ time t, x_target @ time 0 → produce x @ time s
        a_s = self.schedule.alpha(s)   # [B,1]
        a_t = self.schedule.alpha(t)   # [B,1]
        sig_s = self.schedule.sigma(s) # [B,1]
        sig_t = self.schedule.sigma(t) # [B,1]

        term1 = (a_s - (sig_s / sig_t) * a_t) * x_target    # [B, D]
        term2 = (sig_s / sig_t) * x_source                 # [B, D]
        return term1 + term2

    def _kernel(self,
                x: torch.Tensor,
                y: torch.Tensor,
                s: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # Pairwise kernel between x and y at times s < t
        eps = 1e-8
        c_out_st = self.parameterization.c_out(s, t)           # [N,1]
        w = torch.abs(1.0 / (c_out_st + eps))                   # [N,1]

        diff = x - y                                           # [N, D]
        dist = torch.norm(diff, p=2, dim=-1).clamp(min=eps)     # [N,]
        w_flat = w.view_as(dist)                               # [N,]
        log_k = -w_flat * dist / float(self.D)                  # [N,]

        mask = (t.view(-1) > s.view(-1))                        # [N,]
        return torch.where(mask, torch.exp(log_k), torch.zeros_like(log_k))  # [N,]

    def _kernel_for_diff(self,
                         difference: torch.Tensor,
                         s: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(difference)
        return self._kernel(difference, zeros, s.unsqueeze(-1), t.unsqueeze(-1))

    def _weight_fn(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # s,t: [N,1]
        t_flat = t.view(-1)                             # [N,]
        a_t = self.schedule.alpha(t).view(-1)           # [N,]
        sig_t = self.schedule.sigma(t).view(-1)         # [N,]
        logSNR_t = self.schedule.lamb(t).view(-1)       # [N,]
        dlogSNR_t = self.schedule.lamb_diff(t).view(-1) # [N,]

        coef = 0.5 * torch.sigmoid(self.b - logSNR_t) * (-dlogSNR_t)  # [N,]
        denom = a_t.pow(2) + sig_t.pow(2)                              # [N,]
        num = a_t.pow(self.a)                                         # [N,]
        return coef * num / denom                                     # [N,]

    def _map_fn(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = (self.eta_max - self.eta_min) / (2 ** self.k)
        eta_t = self.schedule.eta(t)                 # [B,1]
        val = eta_t - eps
        inv = self.schedule.eta_inv(val)             # [B,1]
        return torch.maximum(s, inv)                  # [B,1]

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D], B must be divisible by M
        returns: scalar joint IMM loss
        """
        B = x.shape[0]
        M = self.M
        assert B % M == 0, f"Batch size ({B}) not divisible by M ({M})"

        G = B // M  # number of groups
        # 1) sample z_prior ~ N(0, σ_data² I) at t=1
        z_prior = self.parameterization.sigma_data * torch.randn_like(x)

        # 2) sample t_single ∈ [tmin, tmax], s_single ∈ [tmin, t_single]
        t_range = self.schedule.tmax - self.schedule.tmin
        t_single = torch.rand((G, 1), device=x.device) * t_range + self.schedule.tmin  # [G,1]
        s_single = torch.rand((G, 1), device=x.device) * (t_single - self.schedule.tmin) + self.schedule.tmin  # [G,1]

        # 3) repeat each M‐times → [B,1]
        t = t_single.repeat_interleave(M, dim=0)  # [B,1]
        s = s_single.repeat_interleave(M, dim=0)  # [B,1]

        # 4) r = map_fn(s, t)
        r = self._map_fn(s, t)  # [B,1]

        # 5) x_t = ddim(z_prior, x, t, 1)
        one_t = torch.ones_like(t)           # [B,1]
        x_t = self._ddim(z_prior, x, t, one_t)  # [B, D]

        # 6) x_r = ddim(x_t, x, r, t)
        x_r = self._ddim(x_t, x, r, t)       # [B, D]

        # 7) f_x_t, f_x_r
        f_x_t = self.forward(x_t, s, t)      # [B, D]
        f_x_r = self.forward(x_r, s, r).detach()  # [B, D]

        # 8) reshape to [G, M, D]
        f_x_t = f_x_t.view(G, M, self.D)     # [G, M, D]
        f_x_r = f_x_r.view(G, M, self.D)     # [G, M, D]

        # 9) pairwise differences (G × M × M × D)
        diff1 = (f_x_t.unsqueeze(1) - f_x_t.unsqueeze(2)).reshape(G*M*M, self.D)
        diff2 = (f_x_r.unsqueeze(1) - f_x_r.unsqueeze(2)).reshape(G*M*M, self.D)
        diff3 = (f_x_t.unsqueeze(1) - f_x_r.unsqueeze(2)).reshape(G*M*M, self.D)

        # 10) build ss, ts, rs arrays of length G*M*M
        ss = s_single.repeat_interleave(M*M, dim=0).view(-1)  # [G*M*M,]
        ts = t_single.repeat_interleave(M*M, dim=0).view(-1)  # [G*M*M,]
        rs = self._map_fn(ss.unsqueeze(-1), ts.unsqueeze(-1)).view(-1)  # [G*M*M,]

        # 11) compute kernel terms
        term1 = self._kernel_for_diff(diff1, ss, ts)  # [G*M*M,]
        term2 = self._kernel_for_diff(diff2, ss, rs)  # [G*M*M,]
        term3 = -2.0 * self._kernel_for_diff(diff3, ss, ts)  # [G*M*M,]

        # 12) weights
        weights = self._weight_fn(ss.unsqueeze(-1), ts.unsqueeze(-1))  # [G*M*M,]

        joint = term1 + term2 + term3  # [G*M*M,]
        return torch.mean(joint * weights)

    def sample(self, n_samples: int, times: torch.Tensor) -> torch.Tensor:
        """
        times: 1D tensor in descending order, e.g. [1.0, 0.9, ..., tmin]
        returns: [n_samples, D] latents at t=tmin
        """
        device = next(self.parameters()).device
        x = self.parameterization.sigma_data * torch.randn((n_samples, self.D), device=device)
        for i in range(len(times) - 1):
            t_val = times[i]
            s_val = times[i+1]
            t_tensor = torch.full((n_samples, 1), t_val, device=device)
            s_tensor = torch.full((n_samples, 1), s_val, device=device)
            x = self.forward(x, s_tensor, t_tensor)
        return x  # [n_samples, D]

def convert_to_display(samples: np.ndarray) -> np.ndarray:
    """
    Tile a batch of images into a square grid.
    samples: [B, H, W, C], B must be a perfect square.
    Returns: [H*grid, W*grid, C].
    """
    B, H, W, C = samples.shape
    cnt = int(math.sqrt(B))
    #assert cnt*cnt == B, "batch size must be a perfect square"
    # To handle non-perfect square, just truncate or pad? 
    # For now, implemented as in notebook.
    if cnt*cnt != B:
        cnt = int(math.sqrt(B))
        samples = samples[:cnt*cnt]
        B = cnt*cnt
        
    # first reshape into (grid_y, grid_x, H, W, C)
    grid = samples.reshape(cnt, cnt, H, W, C)
    # swap axes to (grid_y, H, grid_x, W, C)
    grid = grid.transpose(0, 2, 1, 3, 4)
    # merge to (grid_y*H, grid_x*W, C)
    grid = grid.reshape(cnt * H, cnt * W, C)
    return grid

def show_reconstructions(model, dataloader, device, n=8):
    """
    Grab one batch from dataloader, run it through model,
    and plot the first n originals vs reconstructions.
    """
    model.eval()
    try:
        # grab a single batch
        imgs, _ = next(iter(dataloader))
        imgs = imgs.to(device)

        # forward pass
        with torch.no_grad():
            z, recon = model(imgs)        # or: recon, mu, logvar = model(imgs)
        # take first n
        origs = imgs[:n].cpu()  # [n,1,28,28]
        recs  = recon[:n].cpu()
        # make two image-grids
        grid_orig = torchvision.utils.make_grid(origs.cpu(), nrow=n)
        grid_recon= torchvision.utils.make_grid(recs.cpu(),  nrow=n)

        # plot
        # Need to check if we can show plot (GUI) or save it.
        # Just logging for now if non-interactive
        print("Reconstructions generated (visualization skipped in script mode).")
        # In a real script, we might save fig.
    except Exception as e:
        print(f"Error in show_reconstructions: {e}")

    model.train()

# ────────────────────────────────────────────────────────────────
#  [NEW] Explicit Reconstruction Step
# ────────────────────────────────────────────────────────────────
def calc_reconstruction_mse(x_recon, x):
    """
    Calculates the Mean Squared Error (MSE) reconstruction loss.
    
    Args:
        x_recon: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        
    Returns:
        mse_loss: Scalar tensor representing the mean squared error.
    """
    return F.mse_loss(x_recon, x, reduction='mean')

# ────────────────────────────────────────────────────────────────
#  Training Loop
# ────────────────────────────────────────────────────────────────
def train(
    dataloader,
    z_dim=2,
    n_epochs=10,
    use_cuda=True,
    print_every=100,
    plot_every=500
):
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    print("Current Device:", device)

    # 10.1) Instantiate IMM (D = z_dim, M = 16)
    parametrization = OtFmEulerFmParameterization(sigma_data=1.0)
    imm_net = ResidualDenoiser(input_dim = z_dim + 2, model_dim = 1048, n_blocks  = 8, output_dim= z_dim) #.to(device)
    imm_model = IMM(
        network=imm_net,
        parameterization=parametrization,
        a=1, b=4, eta_max=160.0, eta_min=1e-8, k=12, M=32,
        D=z_dim
    ).to(device)

    # 10.2) Instantiate the VAE (Encoder + Decoder)
    model = Model(z_dim).to(device)

    # 10.3) Two optimizers: VAE params and IMM params
    optimizer_vae = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_imm = torch.optim.Adam(imm_model.parameters(), lr=1e-4)

    step_idx = 0
    for epoch in range(n_epochs):
        model.train()
        imm_model.train()

        for images, _labels in dataloader:
            step_idx += 1

            # ─── Zero gradients for both optimizers ───
            optimizer_vae.zero_grad()
            optimizer_imm.zero_grad()

            # ─── Move batch to device ───
            x = images.to(device)  # [B,3,32,32]

            # ─── Forward pass through VAE ───
            z, x_recon = model(x)  # z: [B, z_dim], x_recon: [B,3,32,32]
            z_det = z.detach()

            # (1) Reconstruction loss (MSE) - [NEW]
            # nll = F.mse_loss(x_recon, x, reduction='mean') # Old way
            nll = calc_reconstruction_mse(x_recon, x) # New explicit way

            # (2) MMD term: sample “true” N(0,1) latents of size [B, z_dim]
            B = z.shape[0]
            # Ensure B is divisible by IMM.M=16
            # assert B % imm_model.M == 0, f"Batch size ({B}) must be divisible by IMM.M ({imm_model.M})"
            # Just skip if not divisible (for last batch)
            if B % imm_model.M != 0:
                continue

            true_samples = torch.randn((B, z_dim), device=device)
            mmd = compute_mmd(true_samples, z)

            n_samples = z.shape[0]           # match batch size
            times = torch.linspace(1.0,
            imm_model.schedule.tmin,
            steps=32,
            device=device)   # e.g. [1.0,0.857,0.714,…,0.0]

            # Now sample z_imm ∈ R^{B×z_dim} using the learned IMM denoiser
            # z_imm = imm_model.sample(n_samples, times)  # [B, z_dim]

            # (b) Prior‐matching MMD between z_enc and z_imm
            # prior_mmd = compute_mmd(z_imm, z)

            # (3) IMM joint‐loss on z
            imm_loss = imm_model.compute_loss(z_det)

            # (4) Total loss = reconstruction + 0.1*mmd + imm_loss
            loss_vae = nll + 1e-5 * mmd 
            loss_imm_net = 1 * imm_loss

            loss_vae.backward(retain_graph=True)   # keep the graph alive so IMM can still use it
            loss_imm_net.backward()

            optimizer_vae.step()
            optimizer_imm.step()

            if step_idx % print_every == 0:
                print(
                    f"[Epoch {epoch+1}/{n_epochs}  Step {step_idx}]  "
                    f"Reconstruction={nll.item():.5f}  MMD={mmd.item():.5f}  IMM={imm_loss.item():.5f}"
                )

    return model, imm_model

# ────────────────────────────────────────────────────────────────
#  Main Execution
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure data dir exists
    if not os.path.exists("./data/CIFAR10"):
        os.makedirs("./data/CIFAR10", exist_ok=True)

    # 11.1) Build a DataLoader
    batch_size = 128 
    # CIFAR10 transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Optional, notebook usedToTensor which is [0,1]
    ])
    
    # Download dataset if needed
    cifar_dataset = CIFAR10("./data/CIFAR10", train=True, download=True, transform=transform)
    cifar_train = DataLoader(
        cifar_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,    # ← drop any partial final batch
        num_workers = 0,
        pin_memory = True
    )

    cifar_test = DataLoader(
        CIFAR10("./data/CIFAR10", train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False
    )

    # 11.2) Train 
    z_dim = 16
    vae_model, imm_model = train(
        dataloader = cifar_train,
        z_dim = z_dim,
        # Reduced epochs for demonstration/verification - user can increase
        n_epochs = 1, 
        use_cuda = True,
        print_every = 50,
        plot_every = 1000
    )

    print("Training complete.")

    # ────────────────────────────────────────────────────────────────
    # [NEW] Evaluation: Reconstruction MSE on Test Set
    # ────────────────────────────────────────────────────────────────
    print("Evaluating Reconstruction MSE on Test Set...")
    device = next(vae_model.parameters()).device
    vae_model.eval()
    
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, _ in cifar_test:
            imgs = imgs.to(device)
            # Forward pass through VAE to get reconstruction
            _, x_recon = vae_model(imgs)
            
            # Calculate MSE using the new function
            batch_mse = calc_reconstruction_mse(x_recon, imgs)
            total_mse += batch_mse.item()
            num_batches += 1

    avg_mse = total_mse / num_batches
    print(f"Test Set Reconstruction MSE: {avg_mse:.5f}")
    
    # Save models just in case
    torch.save(vae_model.state_dict(), "vae_cifar10_mse.pth")
    torch.save(imm_model.state_dict(), "imm_cifar10_mse.pth")
    print("Models saved.")
