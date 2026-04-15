
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
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
    def __init__(self, input_dim, model_dim=1024, n_blocks=8, output_dim=None):
        super().__init__()
        output_dim = output_dim or (input_dim - 2) # Heuristic default
        self.project_in  = nn.Sequential(nn.Linear(input_dim, model_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(model_dim) for _ in range(n_blocks)])
        self.project_out = nn.Sequential(nn.LayerNorm(model_dim), nn.SiLU(), nn.Linear(model_dim, output_dim))

    def forward(self, x):
        h = self.project_in(x)
        for block in self.blocks:
            h = block(h)
        return self.project_out(h)

# -----------------------------------------------------------
# DDPM
# -----------------------------------------------------------
class DDPM(nn.Module):
    def __init__(self, network: nn.Module, z_dim: int, n_steps: int = 1000, device='cuda'):
        super().__init__()
        self.network = network
        self.z_dim = z_dim
        self.n_steps = n_steps
        self.device = device
        
        self.betas = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

    def compute_loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=self.device)
        noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(b, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1)
        
        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        
        # Embed time
        t_emb = self.time_mlp(t.float())
        inp = torch.cat([xt, t_emb], dim=-1)
        
        pred_noise = self.network(inp)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples, steps=None):
        x = torch.randn((n_samples, self.z_dim), device=self.device)
        
        # Determine timeline with stride
        total_steps = self.n_steps
        if steps is None: steps = total_steps
        
        # Create list of time steps [999, ..., 0] possibly strided
        # e.g. if steps=100, then [990, 980, ..., 0] roughly
        stride = total_steps // steps
        time_seq = list(reversed(range(0, total_steps, stride)))
        
        # Add 0 if not included? No, range covers it if divisible.
        
        for i, t_val in enumerate(time_seq):
            t = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)
            
            # Predict Noise
            t_emb = self.time_mlp(t.float())
            inp = torch.cat([x, t_emb], dim=-1)
            pred_noise = self.network(inp)
            
            # Subsampling Params (DDIM-like View)
            # t_prev calculation
            t_prev_val = t_val - stride
            
            # Alphas
            alpha_bar_t = self.alphas_cumprod[t_val]
            alpha_bar_prev = self.alphas_cumprod[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0, device=self.device)
            
            # Pred x0 (denoised)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            
            # Direction to x_t-1
            # Standard DDPM Variance (eta=1.0)
            # sigma_t = sqrt((1-alpha_bar_prev)/(1-alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))
            # But normally we just use fixed variances for simplicity in simple implementation.
            # Let's use the generalized step update:
            # x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + sqrt(1 - alpha_bar_prev - sigma^2) * pred_noise + sigma * noise
            
            # For DDPM behavior (stochastic), eta=1.0
            eta = 1.0
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            
            # Noise
            noise = torch.randn_like(x) if t_prev_val >= 0 else 0
            
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x

# -----------------------------------------------------------
# IMM (Simplified for Project)
# -----------------------------------------------------------
class IMM(nn.Module):
    def __init__(self, network, D=64):
        super().__init__()
        self.network = network
        self.D = D
        # Placeholder for full IMM logic: For this project stage, we can implement 
        # a basic version or rely on the complex 'imm_utils.py' if the user wants strictly that.
        # Given the complexity, I will import the provided 'imm_utils' logic but wrapped cleanly here
        # or implement the core "Iterative Moment Matching" logic if requested.
        # For now, I will use a simplified Flow Matching structure which IMM generalizes.
        pass
        
    # NOTE: Since IMM implementation in the provided codebase was very complex (custom schedulers etc.),
    # I will adapt the 'imm_utils.py' logic into the training loop directly or import it.
    # To keep this file clean, I'll rely on importing the logic in stage2.py or implement the necessary wrappers here.
    # Let's import the robust classes from the existing imm_utils to avoid re-writing 300 lines of math.
