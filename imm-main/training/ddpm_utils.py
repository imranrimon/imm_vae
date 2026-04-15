import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDPM(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 z_dim: int,
                 n_steps: int = 1000,
                 beta_min: float = 1e-4,
                 beta_max: float = 0.02,
                 device: str = 'cuda'):
        super().__init__()
        self.network = network
        self.z_dim = z_dim
        self.n_steps = n_steps
        self.device = device
        
        # Define the linear noise schedule
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, ...] for broadcasting purposes.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def compute_loss(self, x_0):
        """
        Computes the DDPM loss: MSE(epsilon, epsilon_theta(x_t, t))
        """
        batch_size = x_0.shape[0]
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()
        
        # 2. Sample noise
        noise = torch.randn_like(x_0)
        
        # 3. Add noise to x_0 -> x_t
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 4. Predict noise using the network
        # Network expects [B, D+other] usually, but here our network signature depends.
        # The existing ResidualDenoiser in imm_utils takes [B, D] (concatenated).
        # We need to adapt the input to the network. 
        # The IMM implementation did: inp = torch.cat([x_scaled, noise_s, noise_t], dim=-1)
        # We should check what `network` expects.
        # Assuming `network` is the ResidualDenoiser which takes `input_dim`.
        # We need to pass time information.
        # Let's assume we can pass time as an extra dimension or embedding.
        # IMPORTANT: The `ResidualDenoiser` from `imm_utils.py` is just an MLP.
        # We need to feed t into it.
        # Standard way is to concat sinusoidal embedding of t.
        
        t_emb = self.get_sinusoidal_embeddings(t, time_dim=64) # Use 64 dim for time
        # If x_0 is z_dim, then x_t is z_dim.
        # We concat x_t and t_emb.
        
        network_input = torch.cat([x_t, t_emb], dim=-1)
        
        noise_pred = self.network(network_input)
        
        # 5. MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Ancestral sampling from x_T to x_0
        """
        # Start from pure noise
        x = torch.randn((n_samples, self.z_dim), device=self.device)
        
        for i in reversed(range(0, self.n_steps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            t_emb = self.get_sinusoidal_embeddings(t, time_dim=64)
            network_input = torch.cat([x, t_emb], dim=-1)
            noise_pred = self.network(network_input)
            
            # Compute mean: 1/sqrt(alpha) * (x - (beta / sqrt(1-alpha_bar)) * epsilon)
            betas_t = self.extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
            
            mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(x)
                posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                # clamp variance to avoid 0
                # posterior_variance_t = torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
                # actually just use fixed sigma = sqrt(beta) or posterior_variance
                # We use posterior variance here
                std = torch.sqrt(posterior_variance_t)
                x = mean + std * noise
            else:
                x = mean
                
        return x

    def get_sinusoidal_embeddings(self, t, time_dim):
        """
        Returns sinusoidal embeddings for timesteps t.
        """
        half_dim = time_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
