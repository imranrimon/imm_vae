
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training.simple_ae import Model
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper as VQVAE
from training.imm_utils import IMM, OtFmEulerFmParameterization, ResidualDenoiser, SimpleDDPM
from training.sampling_utils import sample_kl_mmd, sample_imm, sample_ddpm

def pairwise_distance(x):
    # x: [N, D]
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, x.t())
    dist = torch.clamp(dist, 0.0, np.inf)
    return torch.sqrt(dist)

def evaluate_diversity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Model
    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if args.model_type == 'vae':
        vae = Model(z_dim=args.z_dim).to(device)
    elif args.model_type == 'nvae':
        input_res = 32 if args.dataset == 'CIFAR10' else 64
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
        # NVAE Wrapper for sampling utils
        class NVAEWrapper(torch.nn.Module):
            def __init__(self, nvae):
                super().__init__()
                self.nvae = nvae
                self.decoder = self
            def forward(self, z): 
                out, _, _ = self.nvae.decoder(z, z_fine=None)
                return out
        
        vae = NVAEWrapper(vae).to(device)
        vae.nvae.load_state_dict(ckpt['vae'])
        vae = NVAEWrapper(vae).to(device)
        vae.nvae.load_state_dict(ckpt['vae'])
    elif args.model_type == 'vqvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = VQVAE(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
        vae.load_state_dict(ckpt['vae'])
    else:
        vae = Model(z_dim=args.z_dim).to(device) # Default fallback
        vae.load_state_dict(ckpt['vae'])
        
    if args.model_type == 'vae':
        res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: res = 32
        elif args.dataset == 'FFHQ': res = 128
        vae = Model(z_dim=args.z_dim, input_res=res).to(device)
        vae.load_state_dict(ckpt['vae'])
    
    vae.eval()
    
    prior_net = None
    sample_fn = None
    
    if args.prior == 'kl' or args.prior == 'mmd':
        sample_fn = sample_kl_mmd
    elif args.prior == 'imm':
        param = OtFmEulerFmParameterization(sigma_data=1.0)
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 2, model_dim=512, n_blocks=6, output_dim=args.z_dim)
        prior_net = IMM(network=denoiser, parameterization=param, D=args.z_dim).to(device)
        prior_net.load_state_dict(ckpt['prior'])
        prior_net.eval()
        sample_fn = sample_imm
    elif args.prior == 'ddpm':
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 1, model_dim=512, n_blocks=6, output_dim=args.z_dim)
        prior_net = SimpleDDPM(network=denoiser, z_dim=args.z_dim).to(device)
        prior_net.load_state_dict(ckpt['prior'])
        prior_net.eval()
        sample_fn = sample_ddpm
        
    # Dataset path logic (strictly for transform, although diversity generated from noise doesn't strictly need real data, 
    # but let's keep it if we reuse it for something, though here we just sample from prior)
    # Actually, evaluate_diversity purely generates samples from VAE+Prior. It doesn't use the dataset!
    # BUT, the script init usually includes data loading logic in case we want to compare or check validity.
    # Logic from other files:
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Path fix just in case it's used or needed for consistency
        dset_root = os.path.join(args.data_root, 'CelebA')
        # dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae':
            t_list.insert(0, transforms.Resize(64))
        transform = transforms.Compose(t_list)
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        # dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)

    # 2. Generate Samples
    print(f"Generating {args.num_samples} samples...")
    batch_size = args.batch_size
    samples_list = []
    
    # Determine steps default if not passed
    sampling_steps = args.steps
    if sampling_steps is None:
            if args.prior == 'imm': sampling_steps = 50
            elif args.prior == 'ddpm': sampling_steps = 1000
    
    with torch.no_grad():
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        for _ in tqdm(range(num_batches)):
            if prior_net:
                if sample_fn.__name__ in ['sample_imm', 'sample_ddpm'] and sampling_steps is not None:
                     s = sample_fn(vae, prior_net, batch_size, device, steps=sampling_steps)
                else:
                     s = sample_fn(vae, prior_net, batch_size, device)
            else:
                s = sample_fn(vae, batch_size, device)
            # Flatten images to vectors [B, D]
            s = s.view(s.shape[0], -1)
            samples_list.append(s)
            
    samples = torch.cat(samples_list, dim=0)[:args.num_samples]
    
    # 3. Compute Metrics
    print("Computing pairwise distances...")
    
    # Subsample if too large for N^2
    max_for_pdist = 2000
    if samples.shape[0] > max_for_pdist:
        indices = torch.randperm(samples.shape[0])[:max_for_pdist]
        samples_sub = samples[indices]
    else:
        samples_sub = samples
        
    dists = pairwise_distance(samples_sub)
    n = samples_sub.shape[0]
    triu_idx = torch.triu_indices(n, n, offset=1)
    mean_dist = dists[triu_idx[0], triu_idx[1]].mean().item()
    
    print(f"Results for {args.prior}:")
    print(f"Mean Pairwise Distance: {mean_dist:.4f}")
    
    import json
    metrics = {
        "diversity": mean_dist
    }
    res_path = os.path.join(os.path.dirname(args.ckpt_path), "metrics_diversity.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prior', type=str, required=True, choices=['kl', 'mmd', 'imm', 'ddpm'], help='Prior type')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--steps', type=int, default=None, help='Number of sampling steps (for IMM/DDPM)')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root') # Added data_root back for consistency

    args = parser.parse_args()
    evaluate_diversity(args)
