
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data import get_dataloader
from src.models.vae import VAE
from src.models.nvae import NVAE

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

def run_reconstruction(model_type='vae', dataset_name='CelebA', z_dim=64):
    print(f"\n--- Visualizing Reconstructions for {model_type.upper()} on {dataset_name} ---\n")
    
    # Load Models
    base_kl = load_base_model(model_type, dataset_name, 'kl', z_dim)
    base_mmd = load_base_model(model_type, dataset_name, 'mmd', z_dim)
    
    # Get Data
    # 8 samples
    loader = get_dataloader(dataset_name, batch_size=8, split='test')
    real_imgs, _ = next(iter(loader))
    real_imgs = real_imgs.to(DEVICE)
    
    rows = []
    labels = []
    
    # Row 1: Real
    rows.append(real_imgs.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5)
    labels.append("Real Images")
    
    # Row 2: KL Recon
    if base_kl:
        with torch.no_grad():
            if model_type == 'vae':
                recon, _, _ = base_kl(real_imgs)
            else:
                out = base_kl(real_imgs)
                recon = out['x_rec']
        rows.append(recon.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5)
        labels.append("KL Reconstruction")
        
    # Row 3: MMD Recon
    if base_mmd:
        with torch.no_grad():
            if model_type == 'vae':
                recon, _, _ = base_mmd(real_imgs)
            else:
                out = base_mmd(real_imgs)
                recon = out['x_rec']
        rows.append(recon.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5)
        labels.append("MMD Reconstruction")
        
    # Plot
    n_samples = 8
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, n_samples, figsize=(15, 2 * n_rows))
    
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
    
    # Output
    out_dir = f"results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    save_file = os.path.join(out_dir, f"reconstructions_{model_type}.png")
    plt.savefig(save_file)
    print(f"\nReconstructions saved to {save_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'nvae'], help='Model Type')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CIFAR10', 'CelebA', 'FFHQ'], help='Dataset Name')
    args = parser.parse_args()
    
    run_reconstruction(args.model, args.dataset)
