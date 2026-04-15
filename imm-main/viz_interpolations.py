
import argparse
import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import numpy as np

from training.simple_ae import Model
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper as VQVAE
from training.imm_utils import IMM, OtFmEulerFmParameterization, ResidualDenoiser, SimpleDDPM
from training.datasets import FlatImageFolder

def visualize_interpolations(args):
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
        
    dataloader = DataLoader(dataset, batch_size=args.num_pairs * 2, shuffle=True, num_workers=0, drop_last=True)
    
    # 2. Load Model
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if args.model_type == 'vae':
        res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: res = 32
        elif args.dataset == 'FFHQ': res = 128
        vae = Model(z_dim=args.z_dim, input_res=res).to(device)
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
         pass
         
    vae.eval()
    
    # 3. Get Pairs
    try:
        x, _ = next(iter(dataloader))
    except StopIteration:
        return
        
    x = x.to(device)
    
    # Encode
    z1, z2 = None, None
    
    if args.model_type == 'vae':
        with torch.no_grad():
            mean, _ = vae.encoder(x)
        z1 = mean[:args.num_pairs]
        z2 = mean[args.num_pairs:2*args.num_pairs]
    elif args.model_type == 'nvae':
        with torch.no_grad():
            mu_g, _, _, _, _, _ = vae.encoder(x)
        z1 = mu_g[:args.num_pairs] # Coarse latents
        z2 = mu_g[args.num_pairs:2*args.num_pairs]
    elif args.model_type == 'vqvae':
        with torch.no_grad():
             z, _ = vae.encoder(x) # Wrapper returns z, dummy
        z1 = z[:args.num_pairs]
        z2 = z[args.num_pairs:2*args.num_pairs]
    
    # Linear Interpolation
    steps = 10
    alphas = torch.linspace(0, 1, steps, device=device)
    
    rows = []
    
    with torch.no_grad():
        for i in range(args.num_pairs):
            row_images = []
            z_start = z1[i].unsqueeze(0)
            z_end = z2[i].unsqueeze(0)
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z_start + alpha * z_end
                
                # Decode
                if args.model_type == 'vae':
                    img_recon = vae.decoder(z_interp)
                elif args.model_type == 'nvae':
                    # NVAE decoder: (z_g, z_f=None) -> samples z_f
                    img_recon, _, _ = vae.decoder(z_interp, z_fine=None)
                elif args.model_type == 'vqvae':
                     img_recon = vae.decoder(z_interp)
                
                # Denormalize [-1, 1] -> [0, 1]
                img_recon = (img_recon + 1) / 2.0
                row_images.append(img_recon)
            
            rows.append(torch.cat(row_images, dim=0))
            
    # Concatenate all rows
    grid = torch.cat(rows, dim=0)
    
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"interpolations_{args.prior}.png")
    utils.save_image(grid, save_path, nrow=steps)
    print(f"Saved interpolations to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prior', type=str, default='vae', help='Prior name for filename')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps (unused for interp but good for tracking)')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')
    parser.add_argument('--num_pairs', type=int, default=8, help='Number of pairs to interpolate')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    visualize_interpolations(args)
