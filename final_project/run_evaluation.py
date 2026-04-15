
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
from torchvision.models import inception_v3
from scipy import linalg

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
    return None

# Try importing LPIPS
try:
    import lpips
    use_lpips = True
except ImportError:
    print("WARNING: lpips library NOT found. Skipping LPIPS.")
    use_lpips = False

def compute_recon_metrics(model, loader, model_type):
    mse_total = 0
    lpips_total = 0
    count = 0
    
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE) if use_lpips else None
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Computing Reconstruction Metrics"):
            x = x.to(DEVICE)
            if model_type == 'vae':
                recon, _, _ = model(x)
            else:
                out = model(x)
                # NVAE returns dictionary
                recon = out['x_rec'] if isinstance(out, dict) else out[1] # Check NVAE return signature if needed
            
            # MSE
            mse_total += F.mse_loss(recon, x, reduction='sum').item()
            
            # LPIPS
            if use_lpips:
                # LPIPS expects [-1, 1]. Data is normalized to [-1, 1].
                # If grayscale (1 channel), repeat to 3
                if x.shape[1] == 1:
                    x_3c = x.repeat(1, 3, 1, 1)
                    recon_3c = recon.repeat(1, 3, 1, 1)
                    lpips_total += lpips_fn(recon_3c, x_3c).sum().item()
                else:
                    lpips_total += lpips_fn(recon, x).sum().item()
            
            count += x.shape[0]
            
    return mse_total / count, (lpips_total / count if use_lpips else 0.0)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.iscomplexobj(covmean.diagonal()):
             covmean = covmean.real
             
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean).real

def compute_local_fid(fake_images, real_loader, max_real_samples=None):
    """
    Computes FID between a batch of fake images and the real dataset loader.
    """
    print("Computing FID (loading InceptionV3)...")
    inception = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
    inception.fc = nn.Identity() 
    inception.eval()
    
    # 1. Real Features
    real_feats = []
    count = 0
    if max_real_samples is None:
        max_real_samples = len(fake_images)

    with torch.no_grad():
        for x, _ in real_loader:
            x = x.to(DEVICE)
            # Map [-1, 1] -> [0, 1] -> ImageNet Norm
            x = x * 0.5 + 0.5 
            x = F.interpolate(x, size=(299, 299), mode='bilinear')
            x = (x - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)) / \
                torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
                
            feat = inception(x)
            if hasattr(feat, 'logits'): feat = feat.logits
            real_feats.append(feat.cpu().numpy())
            count += x.shape[0]
            if count >= max_real_samples: break
            
    real_feats = np.concatenate(real_feats, axis=0)[:max_real_samples]
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    
    # 2. Fake Features
    # fake_images might be on CPU to save memory. 
    # Process in batches to avoid OOM and device mismatch.
             
    fake_feats = []
    BATCH = 32
    
    # Pre-define constants on DEVICE
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
    
    with torch.no_grad():
        for i in range(0, len(fake_images), BATCH):
            b = fake_images[i:i+BATCH].to(DEVICE)
            
            # Preprocess on GPU
            b = b * 0.5 + 0.5
            b = F.interpolate(b, size=(299, 299), mode='bilinear')
            b = (b - mean) / std
            
            f = inception(b)
            if hasattr(f, 'logits'): f = f.logits
            fake_feats.append(f.cpu().numpy())
            
    fake_feats = np.concatenate(fake_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_fake = np.cov(fake_feats, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value

def generate_tsne(latents, labels, title, save_path):
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        indices = labels == lbl
        plt.scatter(latents_2d[indices, 0], latents_2d[indices, 1], label=f"Class {lbl}", s=10, alpha=0.6)
    
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def generate_tsne_comparison(real_z, fake_z, title, save_path):
    print(f"Running t-SNE Comparison for {title}...")
    # Limit samples for speed
    n = min(len(real_z), len(fake_z), 1000)
    real = real_z[:n]
    fake = fake_z[:n].detach().cpu().numpy() if isinstance(fake_z, torch.Tensor) else fake_z[:n]
    
    # 3. Random Gaussian (Prior Reference)
    # Assuming z_dim matches real.shape[1]
    gaussian = np.random.randn(n, real.shape[1])
    
    combined = np.vstack([real, fake, gaussian])
    labels = np.array([0]*n + [1]*n + [2]*n)
    
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    # Plot Real
    plt.scatter(embedded[labels==0, 0], embedded[labels==0, 1], c='blue', label='Real Data', s=10, alpha=0.3)
    # Plot Gaussian
    plt.scatter(embedded[labels==2, 0], embedded[labels==2, 1], c='gray', label='Standard Normal', s=10, alpha=0.2)
    # Plot Fake
    plt.scatter(embedded[labels==1, 0], embedded[labels==1, 1], c='red', label='Learned Prior', s=10, alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def extract_latents(model, loader, model_type, num_samples=2000):
   # ... (Existing Body is fine, I will use reuse it) ...
    latents = []
    labels = []
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            if model_type == 'vae':
                mu, _ = model.encoder(x)
                z = mu
            else:
                mu, _, _, _, _, _ = model.encoder(x)
                z = mu
            
            latents.append(z.cpu().numpy())
            # Handle labels (CelebA y is [B, 40], take first attr)
            if len(y.shape) > 1: y = y[:, 20] # Male attribute
            labels.append(y.cpu().numpy())
            
            count += x.shape[0]
            if count >= num_samples: break
            
    return np.concatenate(latents)[:num_samples], np.concatenate(labels)[:num_samples]

def generate_batched(n_samples, batch_size, z_dim, model_type, base_model, prior_model=None, prior_method='gaussian', steps=None):
    all_imgs = []
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    # Pre-calc times for IMM if needed
    imm_times = None
    if prior_method == 'imm' and steps is not None:
         imm_times = torch.linspace(0, 1, steps+1).to(DEVICE)
         
    with torch.no_grad():
        for i in range(num_batches):
            current_bs = min(batch_size, n_samples - i*batch_size)
            
            # 1. Generate Latents
            if prior_method == 'gaussian':
                z = torch.randn(current_bs, z_dim).to(DEVICE)
            elif prior_method == 'ddpm':
                z = prior_model.sample(current_bs, steps=steps)
            elif prior_method == 'imm':
                z = prior_model.sample(current_bs, times=imm_times)
            
            # 2. Decode
            if model_type == 'vae':
                imgs = base_model.decode(z)
            else:
                imgs = base_model.decoder(z, z_fine=None)
                if isinstance(imgs, tuple): imgs = imgs[0]
            
            all_imgs.append(imgs.cpu()) # Move to CPU
            
    return torch.cat(all_imgs, dim=0)

def evaluate_all(model_type='vae', dataset_name='CIFAR10', z_dim=64, num_samples=500):
    print(f"\n=== Evaluating {model_type.upper()} on {dataset_name} with {num_samples} samples ===\n")
    
    # Output Directory
    out_dir = f"results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Data
    test_loader = get_dataloader(dataset_name, batch_size=128, split='test')
    
    results = []


    # Get Real Latents Reference for t-SNE comparison
    # We need a base model to extract them. Since we iterate, we'll extract inside loop or lazy load.
    # Actually, we can just load the KL base once for reference purposes if we want consistent real latents.
    # But let's stick to the loop flow.
    
    # --- 1. Base Model Evaluations (MSE & t-SNE) ---
    real_latents_cache = {} # Cache per method/base if needed, or just per base model instance

    for method in ['kl', 'mmd']:
        base = load_base_model(model_type, dataset_name, method, z_dim)
        if not base: continue
        
        # Rec Metrics (MSE + LPIPS)
        mse, val_lpips = compute_recon_metrics(base, test_loader, model_type)
        rmse = np.sqrt(mse)
        print(f"[{method.upper()}] Test MSE: {mse:.6f} | RMSE: {rmse:.6f} | LPIPS: {val_lpips:.6f}")
        
        # Generation Metrics (FID)
        imgs_gen = generate_batched(num_samples, 256, z_dim, model_type, base, prior_method='gaussian')
        
        try:
             fid = compute_local_fid(imgs_gen, test_loader, max_real_samples=num_samples)
             print(f"[{method.upper()}] Generation FID: {fid:.4f}")
        except Exception as e:
             fid = -1.0
             print(f"[{method.upper()}] FID Failed: {e}")
        
        # t-SNE (Class Clusters)
        z, y = extract_latents(base, test_loader, model_type)
        real_latents_cache[method] = z # Store for Stage 2 comparison
        
        tsne_path = os.path.join(out_dir, f"tsne_{model_type}_{method}.png")
        generate_tsne(z, y, f"{model_type.upper()}-{method.upper()} Latents (Class Structure)", tsne_path)
        
        results.append({
            "Method": f"{method.upper()}-Base",
            "Steps": "-",
            "MSE": mse,
            "RMSE": rmse,
            "LPIPS": val_lpips,
            "FID": fid
        })

    # --- 2. Generative Evaluations (FID & t-SNE vs Real) ---
    base_kl = load_base_model(model_type, dataset_name, 'kl', z_dim)
    real_z_ref = real_latents_cache.get('kl', None)
    if base_kl is not None and real_z_ref is None:
         real_z_ref, _ = extract_latents(base_kl, test_loader, model_type)

    # DDPM
    ddpm = load_prior_model(model_type, dataset_name, 'ddpm', z_dim)
    if base_kl and ddpm:
        ddpm_steps = [100, 500, 1000]
        # t-SNE Comparison (Generate once with reasonable steps for plot)
        print("Generating DDPM t-SNE comparison...")
        z_tsne = ddpm.sample(n_samples=500, steps=100).detach().cpu().numpy()
        tsne_path = os.path.join(out_dir, f"tsne_{model_type}_ddpm_vs_real.png")
        generate_tsne_comparison(real_z_ref, z_tsne, f"DDPM Generated vs Real Latents", tsne_path)

        for s in ddpm_steps:
            print(f"Evaluating DDPM ({s} steps)...")
            imgs_gen = generate_batched(num_samples, 256, z_dim, model_type, base_kl, prior_model=ddpm, prior_method='ddpm', steps=s)
            
            try:
                 fid = compute_local_fid(imgs_gen, test_loader, max_real_samples=num_samples)
                 print(f"[DDPM-{s}] FID: {fid:.4f}")
            except Exception as e:
                 print(f"[DDPM-{s}] FID Failed: {e}")
                 fid = -1.0
            results.append({"Method": "DDPM", "Steps": s, "MSE": "-", "RMSE": "-", "LPIPS": "-", "FID": fid})
        
    # IMM
    imm = load_prior_model(model_type, dataset_name, 'imm', z_dim)
    if base_kl and imm:
        # t-SNE Comparison (Use fewer steps like 10 for quick visualization?)
        # Or maybe the best one (64)? Let's use 32 as balanced.
        print("Generating IMM t-SNE comparison...")
        times = torch.linspace(0, 1, 33).to(DEVICE)
        z_tsne = imm.sample(n_samples=500, times=times).detach().cpu().numpy()
        tsne_path = os.path.join(out_dir, f"tsne_{model_type}_imm_vs_real.png")
        generate_tsne_comparison(real_z_ref, z_tsne, f"IMM Generated vs Real Latents", tsne_path)

        for s in [8, 16, 32, 64]:
            print(f"Evaluating IMM ({s} steps)...")
            imgs_gen = generate_batched(num_samples, 256, z_dim, model_type, base_kl, prior_model=imm, prior_method='imm', steps=s)
            
            try:
                 fid = compute_local_fid(imgs_gen, test_loader, max_real_samples=num_samples)
                 print(f"[IMM-{s}] FID: {fid:.4f}")
            except Exception as e:
                 print(f"[IMM-{s}] FID Failed: {e}")
                 fid = -1.0
            results.append({"Method": "IMM", "Steps": s, "MSE": "-", "RMSE": "-", "LPIPS": "-", "FID": fid})

    # --- Save Table ---
    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df)
    csv_path = os.path.join(out_dir, f"evaluation_{model_type}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed metrics to {csv_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'nvae'], help='Model Type')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CelebA', 'FFHQ'], help='Dataset Name')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples for FID generation')
    args = parser.parse_args()
    
    evaluate_all(args.model, args.dataset, num_samples=args.samples)
