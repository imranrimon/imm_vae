
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.datasets import FlatImageFolder

def compute_active_units(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Data
    # 1. Load Data
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dset_root = os.path.join(args.data_root, 'CelebA')
        dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae': # VAE expects 64
             t_list.insert(0, transforms.Resize(64))
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
    elif args.dataset == 'FashionMNIST':
        t_list = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform = transforms.Compose(t_list)
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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load Model
    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if args.model_type == 'vae':
        res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: res = 32
        elif args.dataset == 'FFHQ': res = 128
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = Model(z_dim=args.z_dim, in_channels=in_channels, input_res=res).to(device)
    elif args.model_type == 'nvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        from training.vqvae_model import VQVAE_Wrapper
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = VQVAE_Wrapper(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
        
    vae.load_state_dict(ckpt['vae'])
    vae.eval()
    
    # 3. Compute Means
    print("Computing posterior means...")
    mus = []
    threshold = 1e-2
    
    count = 0
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            
            if args.model_type == 'nvae':
                 # NVAE: extract Global Latent mu
                 # forward() returns a dict with 'mu_g'
                 out = vae(x)
                 mu = out['mu_g'] # [B, D]
                 mus.append(mu.cpu())
                 
            elif args.model_type == 'vqvae':
                 # VQVAE: encoder returns "z" (continuous pre-quantized) and dummy logvar
                 z, _ = vae.encoder(x)
                 mus.append(z.cpu())
                 
            else: # VAE
                 mu, _ = vae.encoder(x)
                 mus.append(mu.cpu())
            
            count += x.shape[0]
            if count >= args.num_samples:
                break
    
    if len(mus) > 0:
        mus = torch.cat(mus, dim=0)[:args.num_samples]
        # mus shape: [N, z_dim]
        # variance across N
        vars = torch.var(mus, dim=0)
        active_units = torch.sum(vars > threshold).item()
        print(f"Active Units ( > {threshold}): {active_units} / {args.z_dim}")
        
        import json
        metrics = {
            "active_units": active_units,
            "total_units": args.z_dim
        }
        res_path = os.path.join(os.path.dirname(args.ckpt_path), "metrics_active_units.json")
        with open(res_path, "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        print("No means computed (NVAE support pending).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae'], help='Model Architecture')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')

    args = parser.parse_args()
    compute_active_units(args)
