
import os
import torch
import torch.nn.functional as F
from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE

def loss_vae_kl(recon, x, mu, logvar, kl_weight=0.00025):
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + kl_weight * kl_loss

def loss_kl_general(mu_q, logvar_q, mu_p, logvar_p):
    # KL(N(mu_q, var_q) || N(mu_p, var_p))
    # logvar is log(sigma^2)
    # 0.5 * sum( logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)^2) / var_p )
    # But wait, dividing by exp(logvar_p) is unstable? No, usually fine.
    
    kl = 0.5 * torch.sum(
        logvar_p - logvar_q - 1 + 
        (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
    )
    return kl / mu_q.shape[0] # Average over batch? In loss_vae_kl it was / x.shape[0]


def loss_vae_mmd(z, z_prior_sample):
    # RBF Kernel MMD
    def compute_kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.view(x_size, dim)
        y = y.view(y_size, dim)
        tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
        tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
        return torch.exp(-kernel_input)

    k_zz = compute_kernel(z, z)
    k_pp = compute_kernel(z_prior_sample, z_prior_sample)
    k_zp = compute_kernel(z, z_prior_sample)
    mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
    return mmd

def train_stage1(args, device):
    print(f"--- Stage 1: Training {args.model.upper()} ({args.method}) on {args.dataset} ---")
    
    # 1. Setup Model
    if args.model == 'vae':
        model = VAE(latent_dim=args.z_dim).to(device)
    elif args.model == 'nvae':
        model = NVAE(z_dim=args.z_dim).to(device)
    else:
        raise ValueError(f"Unknown model {args.model}")

    # 2. Setup Data
    loader = get_dataloader(args.dataset, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # KL Weight Config
    kl_w = args.kl_weight if args.kl_weight is not None else 0.00025
    print(f"Using KL Weight: {kl_w}")
    
    save_path = f"checkpoints/{args.model}_{args.dataset}_{args.method}_stage1.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    best_loss = float('inf')
    
    val_loader = get_dataloader(args.dataset, batch_size=128, split='test')
    
    for epoch in range(args.epochs):
        # --- TRAIN ---
        model.train()
        epoch_loss = 0
        steps = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            # Loss Calculation (Shared logic would be cleaner, but keeping inline for now)
            if args.model == 'vae':
                recon, mu, logvar = model(x)
                if args.method == 'kl':
                    loss = loss_vae_kl(recon, x, mu, logvar, kl_weight=kl_w)
                elif args.method == 'mmd':
                    z = model.reparameterize(mu, logvar)
                    z_prior = torch.randn_like(z)
                    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
                    mmd_loss = loss_vae_mmd(z, z_prior)
                    loss = recon_loss + 10.0 * mmd_loss
            else: # NVAE
                out = model(x)
                recon_loss = F.mse_loss(out['x_rec'], x, reduction='sum') / x.shape[0]
                if args.method == 'kl':
                    # 1. Global KL (vs Standard Normal)
                    mu_g, log_g = out['mu_g'], out['log_g']
                    kl_g = -0.5 * torch.sum(1 + log_g - mu_g.pow(2) - log_g.exp()) / x.shape[0]
                    
                    # 2. Fine KL (vs Learned Prior)
                    mu_f, log_f = out['mu_f'], out['log_f']
                    mu_p_f, log_p_f = out['mu_prior_f'], out['log_prior_f']
                    kl_f = loss_kl_general(mu_f, log_f, mu_p_f, log_p_f)
                    
                    # Total Loss (weight KL small initially? Standard NVAE uses spectral regularization or warm-up. 
                    # For simplicity, we use fixed small weight 0.00025 like VAE or slightly higher)
                    loss = recon_loss + kl_w * (kl_g + kl_f)
                elif args.method == 'mmd':
                    mu_g, log_g = out['mu_g'], out['log_g']
                    std_g = torch.exp(0.5 * log_g)
                    z_g = mu_g + torch.randn_like(mu_g) * std_g
                    z_prior = torch.randn_like(z_g)
                    mmd_loss = loss_vae_mmd(z_g, z_prior)
                    
                    # Fine KL (vs Learned Prior) - Essential for stability
                    mu_f, log_f = out['mu_f'], out['log_f']
                    mu_p_f, log_p_f = out['mu_prior_f'], out['log_prior_f']
                    kl_f = loss_kl_general(mu_f, log_f, mu_p_f, log_p_f)
                    
                    loss = recon_loss + 10.0 * mmd_loss + kl_w * kl_f

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        
        train_loss = epoch_loss / steps
        
        # --- VALIDATION ---
        model.eval()
        val_loss_sum = 0
        val_steps = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                if args.model == 'vae':
                    recon, mu, logvar = model(x)
                    if args.method == 'kl':
                        val_step_loss = loss_vae_kl(recon, x, mu, logvar, kl_weight=kl_w)
                    elif args.method == 'mmd':
                        z = model.reparameterize(mu, logvar)
                        z_prior = torch.randn_like(z)
                        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
                        mmd_loss = loss_vae_mmd(z, z_prior)
                        val_step_loss = recon_loss + 10.0 * mmd_loss
                else: # NVAE
                    out = model(x)
                    recon_loss = F.mse_loss(out['x_rec'], x, reduction='sum') / x.shape[0]
                    if args.method == 'kl':
                        mu_g, log_g = out['mu_g'], out['log_g']
                        kl_g = -0.5 * torch.sum(1 + log_g - mu_g.pow(2) - log_g.exp()) / x.shape[0]
                        
                        mu_f, log_f = out['mu_f'], out['log_f']
                        mu_p_f, log_p_f = out['mu_prior_f'], out['log_prior_f']
                        kl_f = loss_kl_general(mu_f, log_f, mu_p_f, log_p_f)
                        
                        val_step_loss = recon_loss + kl_w * (kl_g + kl_f)
                    elif args.method == 'mmd':
                        mu_g, log_g = out['mu_g'], out['log_g']
                        std_g = torch.exp(0.5 * log_g)
                        z_g = mu_g + torch.randn_like(mu_g) * std_g
                        mmd_loss = loss_vae_mmd(z_g, torch.randn_like(z_g))
                        
                        mu_f, log_f = out['mu_f'], out['log_f']
                        mu_p_f, log_p_f = out['mu_prior_f'], out['log_prior_f']
                        kl_f = loss_kl_general(mu_f, log_f, mu_p_f, log_p_f)
                        
                        val_step_loss = recon_loss + 10.0 * mmd_loss + kl_w * kl_f
                
                val_loss_sum += val_step_loss.item()
                val_steps += 1
        
        val_loss = val_loss_sum / val_steps
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    print(f"Saved Base Model to {save_path}")
