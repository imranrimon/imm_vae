
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from training.simple_ae import Model
from training.nvae_model import NVAE
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper as VQVAE
from training.imm_utils import IMM, OtFmEulerFmParameterization, ResidualDenoiser, SimpleDDPM
from training.datasets import FlatImageFolder

# Try importing LPIPS
try:
    import lpips
    use_lpips = True
except ImportError:
    print("WARNING: lpips library NOT found. Skipping LPIPS calculation.")
    use_lpips = False

def evaluate_reconstruction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Path fix
        dset_root = os.path.join(args.data_root, 'CelebA')
        dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae':
            t_list.insert(0, transforms.Resize(64))
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
    elif args.dataset == 'FashionMNIST':
       # Resize 32
       transform = transforms.Compose([
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
       ])
       dset_root = os.path.join(args.data_root, 'FashionMNIST')
       dataset = datasets.FashionMNIST(root=dset_root, train=False, download=True, transform=transform)

    elif args.dataset == 'FFHQ':
       transform = transforms.Compose([
           transforms.Resize(128),
           transforms.CenterCrop(128),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
       dset_root = os.path.join(args.data_root, 'FFHQ')
       dataset = FlatImageFolder(root=dset_root, transform=transform)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    # 2. Load Model
    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if args.model_type == 'vae':
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        # Resolution logic
        res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: res = 32
        elif args.dataset == 'FFHQ': res = 128
        
        vae = Model(z_dim=args.z_dim, in_channels=in_channels, input_res=res).to(device)
    elif args.model_type == 'nvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = VQVAE(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
        
    if 'vae' in ckpt:
         vae.load_state_dict(ckpt['vae'])
    else:
         # Maybe direct load if saved differently? Assume strict format.
         pass
         
    vae.eval()
    
    # 3. Metrics
    mse_loss_fn = torch.nn.MSELoss(reduction='none')
    lpips_loss_fn = None
    if use_lpips:
        lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        
    total_mse = 0.0
    total_lpips = 0.0
    count = 0
    
    print(f"Evaluating Reconstruction ({len(dataset)} items)...")
    
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            B = x.shape[0]
            
            x_rec = None
            if args.model_type == 'vae':
                mean, logvar = vae.encoder(x)
                x_rec = vae.decoder(mean)
            elif args.model_type == 'nvae':
                out = vae(x)
            elif args.model_type == 'nvae':
                out = vae(x)
                x_rec = out['x_rec'] # NVAE forward returns reparameterized reconstructions by default
            elif args.model_type == 'vqvae':
                 _, x_rec, _, _ = vae(x) # VQVAE returns (quantized, x_rec, ...)
            
            # MSE
            mse = mse_loss_fn(x_rec, x).view(B, -1).mean(dim=1).sum()
            total_mse += mse.item()
            
            # LPIPS
            if use_lpips:
                # x_rec, x are [-1, 1], matches LPIPS expectation
                # If grayscale, repeat channels for LPIPS (which usually expects 3)
                x_rec_lpips = x_rec
                x_lpips = x
                if x.shape[1] == 1:
                     x_rec_lpips = x_rec.repeat(1, 3, 1, 1)
                     x_lpips = x.repeat(1, 3, 1, 1)
                     
                val_lpips = lpips_loss_fn(x_rec_lpips, x_lpips).view(B).sum()
                total_lpips += val_lpips.item()
                
            count += B
            
            if args.max_items and count >= args.max_items:
                break
                
    avg_mse = total_mse / count
    avg_lpips = total_lpips / count if use_lpips else -1.0
    
    print(f"Results for {args.ckpt_path}:")
    print(f"MSE: {avg_mse:.6f}")
    if use_lpips:
        print(f"LPIPS: {avg_lpips:.6f}")
        
    # Save results
    import json
    metrics = {
        "mse": avg_mse,
        "lpips": avg_lpips if use_lpips else None
    }
    
    res_path = os.path.join(os.path.dirname(args.ckpt_path), "metrics_recon.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved reconstruction metrics to {res_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_items', type=int, default=None, help='Max items to evaluate')
    
    args = parser.parse_args()
    evaluate_reconstruction(args)
