
import torch
from training.imm_utils import IMM, SimpleDDPM, OtFmEulerFmParameterization, ResidualDenoiser
from training.loss import DDPMLoss

@torch.no_grad()
def sample_kl_mmd(vae, n_samples, device):
    vae.eval()
    z = torch.randn(n_samples, vae.z_dim, device=device)
    samples = vae.decoder(z)
    return samples

@torch.no_grad()
def sample_imm(vae, imm_prior, n_samples, device, steps=50):
    vae.eval()
    imm_prior.eval()
    
    # IMM sampling involves solving the ODE/SDE from noise to data distribution of latents
    # The IMM class has a sample method: .sample(n_samples, times)
    # times should be a tensor of decreasing time steps from 1.0 to epsilon
    
    times = torch.linspace(1.0, 1e-4, steps, device=device)
    z_samples = imm_prior.sample(n_samples, times)
    
    samples = vae.decoder(z_samples)
    return samples

@torch.no_grad()
def sample_ddpm(vae, ddpm_prior, n_samples, device, steps=1000, schedule="cosine"):
    vae.eval()
    ddpm_prior.eval()
    
    import math

    # Continuous Cosine Schedule matching DDPMLoss
    def get_alpha_bar(t):
        s = 0.008
        return torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2 / (math.cos(s / (1 + s) * math.pi / 2) ** 2)

    # Start from pure noise
    z = torch.randn(n_samples, vae.z_dim, device=device)
    
    # Iterate from t=1.0 down to t=0
    # We discretize [0, 1] into 'steps' segments.
    # Corresponds to t_i = i / steps.
    # Start loop at i=steps (t=1), go down to i=1 (t=1/steps).
    # Target is x_0 at i=0.
    
    for i in reversed(range(1, steps + 1)):
        # Current time t
        t_float = i / steps
        t_prev_float = (i - 1) / steps
        
        t_tensor = torch.full((n_samples, 1), t_float, device=device)
        
        # Helper: alpha_bar variables
        # Note: We need standard scalar/tensor math here.
        
        ab_t = get_alpha_bar(torch.tensor(t_float))
        ab_prev = get_alpha_bar(torch.tensor(t_prev_float))
        
        # Derived beta_t and alpha_t for this discrete step
        # alpha_bar_t = alpha_bar_prev * alpha_t
        # => alpha_t = alpha_bar_t / alpha_bar_prev
        # Guard against zero division if t is tiny, though cosine schedule is safe >0 usually.
        # But at t=0, ab=1.
        
        alpha_t = ab_t / ab_prev
        beta_t = 1 - alpha_t
        
        # Posterior variance (sigma_t^2)
        # sigma_t^2 = beta_t * (1 - ab_prev) / (1 - ab_t) (for posterior q(x_{t-1}|x_t, x_0))
        # Or simply beta_t for standard DDPM (sigma_t = sqrt(beta_t)). Both work.
        # We'll use the proper posterior variance.
        sigma_t = torch.sqrt(beta_t * (1 - ab_prev) / (1 - ab_t))
        
        # Predict noise
        e_theta = ddpm_prior(z, t_tensor)
        
        # Update z
        # mean = 1/sqrt(alpha_t) * (z - (beta_t / sqrt(1 - ab_t)) * e_theta)
        # We need tensors for broadcasting if we used scalar math above.
        
        recip_sqrt_alpha = 1 / torch.sqrt(alpha_t)
        noise_coeff = beta_t / torch.sqrt(1 - ab_t)
        
        mean = recip_sqrt_alpha * (z - noise_coeff * e_theta)
        
        if i > 1:
            noise = torch.randn_like(z)
            z = mean + sigma_t * noise
        else:
            z = mean
            
    samples = vae.decoder(z)
    return samples
