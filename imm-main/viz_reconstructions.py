
import argparse
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper
from training.datasets import FlatImageFolder

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing Reconstructions: {args.dataset} | {args.model_type} | {args.ckpt_path}")

    # 1. Dataset Setup
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dset_root = os.path.join(args.data_root, 'CelebA')
        val_dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
        in_channels = 3
        input_res = 64
        
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae':
             t_list.insert(0, transforms.Resize(64))
             input_res = 64
        else:
             input_res = 32
        
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        val_dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
        in_channels = 3

    elif args.dataset == 'FashionMNIST':
        t_list = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'FashionMNIST')
        val_dataset = datasets.FashionMNIST(root=dset_root, train=False, download=True, transform=transform)
        in_channels = 1
        output_res = 32
        input_res = 32
        
    elif args.dataset == 'FFHQ':
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dset_root = os.path.join(args.data_root, 'FFHQ')
        val_dataset = FlatImageFolder(root=dset_root, transform=transform)
        in_channels = 3
        input_res = 128

    # Fixed seed for data loader shuffling (though we won't shuffle for consistency)
    torch.manual_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=args.n_imgs, shuffle=False, num_workers=0)

    # 2. Model Setup
    # Use args.z_dim to match training
    if args.model_type == 'vae':
        vae = Model(z_dim=args.z_dim, in_channels=in_channels, input_res=input_res).to(device)
    elif args.model_type == 'nvae':
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        vae = VQVAE_Wrapper(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    vae.load_state_dict(checkpoint['vae'])
    vae.eval()

    # 3. Reconstruct
    x, _ = next(iter(val_loader))
    x = x.to(device)
    
    with torch.no_grad():
        if args.model_type == 'vqvae':
            _, x_rec, _, _ = vae(x)
        elif args.model_type == 'vae':
            _, x_rec, _, _ = vae(x)
        elif args.model_type == 'nvae':
            out = vae(x)
            x_rec = out['x_rec']

    # 4. Save
    # Denormalize: [-1,1] -> [0,1]
    x = (x + 1) / 2
    x_rec = (x_rec + 1) / 2
    
    # Grid: Top is Original, Bottom is Recon
    # Interleave: [Orig1, Recon1, Orig2, Recon2...] or just stacked grids?
    # Stacked grids is clearer.
    
    grid_x = make_grid(x, nrow=8)
    grid_rec = make_grid(x_rec, nrow=8)
    
    # Concatenate vertically
    # grid shape [C, H, W]
    combined = torch.cat([grid_x, grid_rec], dim=1) # Dim 1 is Height
    
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"recon_{args.dataset}_{args.model_type}_{args.prior}.png")
    save_image(combined, save_path)
    print(f"Saved reconstructions to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--prior', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='e:/imm_vae/data')
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--n_imgs', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=128)
    args = parser.parse_args()
    main(args)
