
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE
from src.models.diffusion import DDPM, ResidualDenoiser
from src.models.imm_core import IMM, OTFMSchedule, OtFmEulerFmParameterization

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_base_model(model_type, dataset_name, method, z_dim=64):
    ckpt = f"checkpoints/{model_type}_{dataset_name}_{method}_stage1.pth"
    if model_type == 'vae':
        model = VAE(latent_dim=z_dim).to(DEVICE)
    else:
        model = NVAE(z_dim=z_dim).to(DEVICE)
        
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded {method.upper()}-Base: {ckpt}")
        model.eval()
        return model
    else:
        print(f"Warning: Checkpoint {ckpt} not found!")
        return None

def load_prior_model(model_type, dataset_name, method, z_dim=64):
    ckpt = f"checkpoints/{model_type}_{dataset_name}_{method}_stage2.pth"
    
    if method == 'ddpm':
        denoiser = ResidualDenoiser(input_dim=z_dim+64, model_dim=512, output_dim=z_dim, n_blocks=6)
        model = DDPM(denoiser, z_dim=z_dim, n_steps=1000, device=DEVICE).to(DEVICE)
    elif method == 'imm':
        denoiser = ResidualDenoiser(input_dim=z_dim+2, model_dim=512, output_dim=z_dim, n_blocks=6)
        param = OtFmEulerFmParameterization(sigma_data=1.0)
        model = IMM(denoiser, param, D=z_dim, M=16).to(DEVICE)
        
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded {method.upper()}-Prior: {ckpt}")
        model.eval()
        return model
    else:
        print(f"Warning: Prior Checkpoint {ckpt} not found!")
        return None

def decode_batch(base_model, z, model_type):
    with torch.no_grad():
        if model_type == 'vae':
            imgs = base_model.decode(z)
        else:
            imgs, _, _ = base_model.decoder(z, z_fine=None)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    return imgs

def run_comparison(model_type='vae', dataset_name='CIFAR10', z_dim=64):
    print(f"\n--- Generating Comparison for {model_type.upper()} on {dataset_name} ---\n")
    
    # 1. Load Everything
    base_kl = load_base_model(model_type, dataset_name, 'kl', z_dim)
    base_mmd = load_base_model(model_type, dataset_name, 'mmd', z_dim)
    
    # Priors (Assumed trained on KL base, as per stage2 defaults)
    prior_ddpm = load_prior_model(model_type, dataset_name, 'ddpm', z_dim)
    prior_imm = load_prior_model(model_type, dataset_name, 'imm', z_dim)
    
    rows = []
    labels = []
    
    n_samples = 8
    
    # --- 1. KL-VAE (Baseline) ---
    if base_kl:
        z = torch.randn(n_samples, z_dim).to(DEVICE) # Simple Gaussian
        imgs = decode_batch(base_kl, z, model_type)
        rows.append(imgs)
        labels.append(f"{model_type.upper()}-KL (Standard)")

    # --- 2. MMD-VAE (Baseline) ---
    if base_mmd:
        z = torch.randn(n_samples, z_dim).to(DEVICE) # Simple Gaussian (MMD enforces this)
        imgs = decode_batch(base_mmd, z, model_type)
        rows.append(imgs)
        labels.append(f"{model_type.upper()}-MMD")
        
    # --- 3. DDPM-VAE (Ours) ---
    if base_kl and prior_ddpm:
         # Custom steps for DDPM as requested
         ddpm_steps = [100, 500, 1000]
         for s in ddpm_steps:
             print(f"Sampling DDPM ({s} steps)...")
             z = prior_ddpm.sample(n_samples, steps=s) 
             imgs = decode_batch(base_kl, z, model_type)
             rows.append(imgs)
             labels.append(f"DDPM Prior ({s})")
        
    # --- 4. IMM-VAE (Ours - Varying Steps) ---
    if base_kl and prior_imm:
        steps_list = [8, 16, 32, 64]
        for s in steps_list:
            print(f"Sampling IMM ({s} steps)...")
            times = torch.linspace(0, 1, s+1).to(DEVICE)
            z = prior_imm.sample(n_samples, times)
            imgs = decode_batch(base_kl, z, model_type)
            rows.append(imgs)
            labels.append(f"IMM Prior ({s})")

    # --- Plotting ---
    if not rows:
        print("No models loaded, nothing to plot.")
        return

    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, n_samples, figsize=(15, 2 * n_rows))
    
    # If only 1 row, axs is 1D array
    if n_rows == 1: axs = axs[None, :]

    for r in range(n_rows):
        for c in range(n_samples):
            ax = axs[r, c]
            ax.imshow(rows[r][c].clip(0, 1))
            ax.axis('off')
            if c == 0:
                ax.text(-0.2, 0.5, labels[r], transform=ax.transAxes, 
                        va='center', ha='right', fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    
    # Create Output Directory
    out_dir = f"results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    save_file = os.path.join(out_dir, f"full_grid_{model_type}.png")
    plt.savefig(save_file)
    print(f"\nComparison saved to {save_file}")
    plt.show()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'nvae'], help='Model Type')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CelebA', 'FFHQ'], help='Dataset Name')
    args = parser.parse_args()
    
    run_comparison(args.model, args.dataset)
