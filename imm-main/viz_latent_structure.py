
import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from training.simple_ae import Model
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper as VQVAE
from training.datasets import FlatImageFolder

def visualize_latent_structure(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        has_labels = True # CelebA has attributes, but dataset object .attr is tensor
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae':
            t_list.insert(0, transforms.Resize(64))
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
        has_labels = True # CIFAR has classes
    elif args.dataset == 'FashionMNIST':
       transform = transforms.Compose([
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
       ])
       dset_root = os.path.join(args.data_root, 'FashionMNIST')
       dataset = datasets.FashionMNIST(root=dset_root, train=False, download=True, transform=transform)
       dset_root = os.path.join(args.data_root, 'FashionMNIST')
       dataset = datasets.FashionMNIST(root=dset_root, train=False, download=True, transform=transform)
       has_labels = True
    elif args.dataset == 'FFHQ':
       transform = transforms.Compose([
           transforms.Resize(128),
           transforms.CenterCrop(128),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
       dset_root = os.path.join(args.data_root, 'FFHQ')
       dataset = FlatImageFolder(root=dset_root, transform=transform)
       has_labels = False # FFHQ has no labels in this flat format

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load Model
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if args.model_type == 'vae':
        res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: res = 32
        elif args.dataset == 'FFHQ': res = 128
        vae = Model(z_dim=args.z_dim, input_res=res).to(device)
    elif args.model_type == 'nvae':
        input_res = 32 if args.dataset == 'CIFAR10' else 64
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = VQVAE(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
        
    if 'vae' in ckpt:
         vae.load_state_dict(ckpt['vae'])
         
    vae.eval()
    
    # 3. Extract Latents
    latents = []
    labels = []
    
    print(f"Extracting latents ({args.num_samples} samples)...")
    count = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            B = x.shape[0]
            
            if args.model_type == 'vae':
                mean, _ = vae.encoder(x)
                z = mean
            elif args.model_type == 'nvae':
                mu_g, _, _, _, _, _ = vae.encoder(x)
                z = mu_g # Use Coarse Latent
            elif args.model_type == 'vqvae':
                z, _ = vae.encoder(x)
            
            latents.append(z.cpu().numpy())
            
            # Labels
            # CelebA y is [B, 40] attributes. We can pick one (e.g. Gender=20, or Male=20)
            # CIFAR y is [B] class index.
            if args.dataset == 'CelebA':
                 # Attribute 20 (Male) usually visible
                 labels.append(y[:, 20].cpu().numpy()) 
            else:
                 labels.append(y.cpu().numpy())
            
            count += B
            if count >= args.num_samples:
                break
                
    latents = np.concatenate(latents, axis=0)[:args.num_samples]
    labels = np.concatenate(labels, axis=0)[:args.num_samples]
    
    # 4. t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42) # n_iter removed
    latents_2d = tsne.fit_transform(latents)
    
    # 5. Plot
    # 5. Plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        indices = labels == lbl
        plt.scatter(latents_2d[indices, 0], latents_2d[indices, 1], label=f"Class {lbl}", s=10, alpha=0.6)
    
    plt.legend()
    
    title_str = f"t-SNE of {args.prior.upper()} Latents ({args.dataset})"
    if args.steps:
        title_str += f" | Steps: {args.steps}"
    plt.title(title_str)
    
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"tsne_{args.prior}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples to visualize')
    parser.add_argument('--prior', type=str, default='vae', help='Prior name for filename')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps for title')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    visualize_latent_structure(args)
