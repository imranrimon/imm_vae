
import os
import torch
import torch.nn as nn
from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE
from src.models.diffusion import DDPM, ResidualDenoiser
# Import IMM Core
from src.models.imm_core import IMM, OTFMSchedule, OtFmEulerFmParameterization
from src.models.meanflow import MeanFlow, MeanFlowNet

def compute_kernel(x, y):
    x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
    x_exp = x.unsqueeze(1).expand(x_size, y_size, dim)
    y_exp = y.unsqueeze(0).expand(x_size, y_size, dim)
    kernel_input = ((x_exp - y_exp)**2).mean(2) / float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x).mean()
    y_kernel = compute_kernel(y, y).mean()
    xy_kernel = compute_kernel(x, y).mean()
    return x_kernel + y_kernel - 2*xy_kernel

def train_stage2(args, device):
    print(f"--- Stage 2: Training {args.method.upper()} Prior on {args.model.upper()}-{args.dataset} ---")
    
    # 1. Load Base Model
    # Determine which base model to load (defaulting to 'kl' if not specified, 
    # but ideally user trains stage 2 on specific base)
    # Let's assume user wants to train Prior on the 'kl' base by default unless logic changes
    # Determine which base model to load
    base_method = args.base_method
    rt = getattr(args, 'run_tag', '') or ''
    base_ckpt = f"checkpoints/{args.model}_{args.dataset}_{base_method}{rt}_stage1.pth"
    
    if args.model == 'vae':
        base_model = VAE(latent_dim=args.z_dim).to(device)
    else:
        base_model = NVAE(z_dim=args.z_dim).to(device)
        
    if not os.path.exists(base_ckpt):
        print(f"Error: Base checkpoint {base_ckpt} not found.")
        return
        
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad = False
    
    # 2. Setup Prior Model
    save_path = f"checkpoints/{args.model}_{args.dataset}_{args.method}{rt}_stage2.pth"
    
    # Loaders needed for sigma calc
    # IMM requires batch size divisible by M (16)
    use_drop_last = (args.method == 'imm')
    loader = get_dataloader(args.dataset, batch_size=128, drop_last=use_drop_last)
    val_loader = get_dataloader(args.dataset, batch_size=128, split='test', drop_last=use_drop_last)

    def _latent_stats():
        print("Calculating robust global latent statistics from base model...")
        z_samples = []
        max_batches = 50
        for i, (xb, _) in enumerate(loader):
            if i >= max_batches: break
            xb = xb.to(device)
            with torch.no_grad():
                if args.model == 'vae':
                    mu, logvar = base_model.encode(xb)
                    z = base_model.reparameterize(mu, logvar)
                else:
                     mu_g, log_g, _, _, _, _ = base_model.encoder(xb)
                     std_g = torch.exp(0.5 * log_g)
                     z = mu_g + torch.randn_like(mu_g) * std_g
                z_samples.append(z)
        z_all = torch.cat(z_samples, dim=0)
        m = z_all.mean(dim=0).detach()
        sd = z_all.std(dim=0).detach()
        print(f"  -> Calculated Global Mean: {m.mean():.4f}, Global Std: {sd.mean():.4f}")
        print(f"  -> Std Range: [{sd.min():.4f}, {sd.max():.4f}]")
        return m, sd

    if args.method == 'ddpm':
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 64, model_dim=512, output_dim=args.z_dim, n_blocks=6)
        prior_model = DDPM(denoiser, z_dim=args.z_dim, n_steps=1000, device=device).to(device)
    elif args.method in ('meanflow', 'meanflow_improved'):
        norm_mean, norm_std = _latent_stats()
        net = MeanFlowNet(z_dim=args.z_dim, model_dim=512, n_blocks=6)
        prior_model = MeanFlow(
            net,
            D=args.z_dim,
            variant='base' if args.method == 'meanflow' else 'improved',
            data_mean=norm_mean,
            data_std=norm_std,
            jvp_free=getattr(args, 'jvp_free', False),
        ).to(device)
    elif args.method == 'imm':
        norm_mean, norm_std = _latent_stats()

        # IMM Setup
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 2, model_dim=512, output_dim=args.z_dim, n_blocks=6)
        # Use sigma_data=1.0 because we normalize manually
        param = OtFmEulerFmParameterization(sigma_data=1.0)
        # Pass normalization stats to IMM
        prior_model = IMM(denoiser, param, D=args.z_dim, M=16, data_mean=norm_mean, data_std=norm_std).to(device)
    else:
        raise ValueError(f"Unknown Stage 2 method: {args.method}")

    if args.method == 'imm':
        # Match notebook learning rate for IMM
        default_lr = 1e-5
    elif args.method in ('meanflow', 'meanflow_improved'):
        default_lr = 1e-4
    else:
        default_lr = 1e-4
    lr = getattr(args, 'lr', None) or default_lr
    print(f"Stage 2 optimizer: Adam(lr={lr})")
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=lr)
    
    best_loss = float('inf')
    last_save_path = save_path.replace(".pth", "_last.pth")
    
    for epoch in range(args.epochs):
        prior_model.train()
        epoch_loss = 0
        steps = 0
        
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                if args.model == 'vae':
                    mu, logvar = base_model.encode(x)
                    z = base_model.reparameterize(mu, logvar)
                else: # NVAE
                    mu_g, log_g, _, _, _, _ = base_model.encoder(x)
                    std_g = torch.exp(0.5 * log_g)
                    z = mu_g + torch.randn_like(mu_g) * std_g
            
            if args.method in ('meanflow', 'meanflow_improved'):
                loss = prior_model.compute_loss(z)
            elif args.method == 'imm':
                # Joint Training Logic: Score Matching + MMD
                imm_loss = prior_model.compute_loss(z)
                
                # Sample from IMM to compute MMD
                # IMM.sample returns denormalized samples matching z space
                times = torch.linspace(1.0, prior_model.schedule.tmin, steps=32, device=device)
                z_imm = prior_model.sample(z.shape[0], times)
                
                prior_mmd = compute_mmd(z_imm, z)
                
                # Combined loss: VAE (frozen here) + IMM (updated by both signals)
                # Notebook uses 1e-4 weight for MMD term in VAE loss which also flows to IMM via z_imm
                loss = imm_loss + 1e-4 * prior_mmd
            else:
                loss = prior_model.compute_loss(z)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        # --- VALIDATION ---
        prior_model.eval()
        val_loss_sum = 0
        val_steps = 0
        
        with torch.no_grad():
             for x, _ in val_loader:
                x = x.to(device)
                
                # Extract Latents (Reuse logic?)
                if args.model == 'vae':
                    mu, logvar = base_model.encode(x)
                    z = base_model.reparameterize(mu, logvar)
                else: # NVAE
                    mu_g, log_g, _, _, _, _ = base_model.encoder(x)
                    std_g = torch.exp(0.5 * log_g)
                    z = mu_g + torch.randn_like(mu_g) * std_g
                
                loss = prior_model.compute_loss(z)
                val_loss_sum += loss.item()
                val_steps += 1
                
        val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
        train_loss = epoch_loss/steps
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
        # Save Last
        torch.save(prior_model.state_dict(), last_save_path)
        
        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(prior_model.state_dict(), save_path)
            print(f"  -> New Best Prior Saved!")
        
    print(f"Saved Best Prior Model to {save_path}")
