
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

# Imports from our local modules
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper
from training.imm_utils import IMM, OtFmEulerFmParameterization, ResidualDenoiser, SimpleDDPM
# Loss classes
from training.loss import KLLoss, MMDLoss, DDPMLoss

class FlatImageFolder(torch.utils.data.Dataset):
    "REMOVED - Imported from training.datasets"
    pass

from training.datasets import FlatImageFolder

def calc_reconstruction_mse(x_recon, x):
    return F.mse_loss(x_recon, x, reduction='mean')

def train(args, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Prior: {args.prior} | Dataset: {args.dataset} | Model: {args.model_type}")

    # 2. Model (Backbone AE / NVAE)
    if args.model_type == 'vae':
        # Standard VAE
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        
        # Determine resolution
        input_res = 64
        if args.dataset in ['CIFAR10', 'FashionMNIST']: input_res = 32
        elif args.dataset == 'FFHQ': input_res = 128
        
        vae = Model(z_dim=args.z_dim, in_channels=in_channels, input_res=input_res).to(device)
    elif args.model_type == 'nvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
        in_channels = 1 if args.dataset == 'FashionMNIST' else 3
        vae = VQVAE_Wrapper(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
    
    # 3. Prior Setup
    prior_net = None
    prior_optimizer = None
    
    if args.prior == 'imm':
        param = OtFmEulerFmParameterization(sigma_data=1.0)
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 2, model_dim=512, n_blocks=6, output_dim=args.z_dim)
        prior_net = IMM(network=denoiser, parameterization=param, D=args.z_dim).to(device)
        prior_optimizer = optim.Adam(prior_net.parameters(), lr=args.lr_prior)
        
    elif args.prior == 'ddpm':
        denoiser = ResidualDenoiser(input_dim=args.z_dim + 128, model_dim=512, n_blocks=6, output_dim=args.z_dim)
        prior_net = SimpleDDPM(network=denoiser, z_dim=args.z_dim, time_dim=128).to(device)
        prior_optimizer = optim.Adam(prior_net.parameters(), lr=args.lr_prior)
        ddpm_criterion = DDPMLoss(noise_schedule="cosine")
        
    elif args.prior == 'kl':
        prior_criterion = KLLoss(kl_weight=args.kl_weight)
        
    elif args.prior == 'mmd':
        prior_criterion = MMDLoss(mmd_weight=args.mmd_weight)

    vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr_vae)

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Resume Logic
    start_epoch = 0
    # Search for latest checkpoint
    if os.path.exists(args.save_dir):
        valid_ckpts = []
        for f in os.listdir(args.save_dir):
            if f.startswith(f"ckpt_{args.prior}_epoch") and f.endswith(".pth"):
                try:
                    # Strict parsing: File must be exactly ckpt_{prior}_epoch{N}.pth
                    # Split exactly by the expected prefix and suffix
                    prefix = f"ckpt_{args.prior}_epoch"
                    suffix = ".pth"
                    if f == prefix + f[len(prefix):-len(suffix)] + suffix:
                        epoch_str = f[len(prefix):-len(suffix)]
                        epoch_num = int(epoch_str)
                        valid_ckpts.append((epoch_num, f))
                except ValueError:
                    continue # Skip files like epoch49_untuned.pth
        
        if valid_ckpts:
            latest_epoch, latest_file = max(valid_ckpts, key=lambda x: x[0])
            latest_ckpt = os.path.join(args.save_dir, latest_file)
            
            print(f"Resuming from checkpoint: {os.path.abspath(latest_ckpt)}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                vae.load_state_dict(checkpoint['vae'])
                if prior_net and checkpoint['prior']:
                    prior_net.load_state_dict(checkpoint['prior'])
                    
                start_epoch = latest_epoch + 1
                print(f"Resuming training at Epoch {start_epoch}")
            except Exception as e:
                 print(f"Failed to load checkpoint {latest_ckpt}: {e}")

    step = 0 # Ideally step should be saved too, but resetting step counter for logging is acceptable if epoch is correct.
    for epoch in range(start_epoch, args.epochs):
        vae.train()
        if prior_net: prior_net.train()
        
        # Safe dataloader iteration
        try:
            iterator = iter(train_loader)
        except Exception as e:
            print(f"Error creating dataloader iterator: {e}")
            break
            
        for batch_idx in range(len(train_loader)):
            try:
                x, _ = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                print(f"Error during batch loading: {e}")
                continue
                
            step += 1
            x = x.to(device)
            
            vae_optimizer.zero_grad()
            if prior_optimizer: prior_optimizer.zero_grad()
            
            # Forward Pass
            z_for_prior = None
            mean, logvar = None, None
            kl_fine = torch.tensor(0.0, device=device)
            
            if args.model_type == 'vqvae':
                # VQ-VAE returns: (quantized, x_rec, vq_loss, perplexity)
                z, x_rec, vq_loss, perplexity = vae(x)
                # For compatibility, we can treat vq_loss as "Prior" loss (commitment)
                prior_loss = vq_loss
                # No mean/logvar
                z_for_prior = z # Quantized latents
                kl_fine = torch.tensor(0.0, device=device)
            elif args.model_type == 'vae':
                z, x_rec, mean, logvar = vae(x)
                z_for_prior = z 
                
            elif args.model_type == 'nvae':
                out = vae(x)
                x_rec = out['x_rec']
                # Coarse Latent for complex prior
                z_for_prior = out['z_g']
                
                # NVAE Coarse KL params (if using KL/MMD prior)
                mean, logvar = out['mu_g'], out['log_g']
                
                # Calculate Fine KL: KL(q(z_f|x) || p(z_f|z_g))
                # p(z_f|z_g) is conditional Gaussian from decoder
                mu_q, logvar_q = out['mu_f'], out['log_f']
                mu_p, logvar_p = out['mu_prior_f'], out['log_prior_f']
                
                # Clamping for stability
                logvar_q = torch.clamp(logvar_q, -10, 10)
                logvar_p = torch.clamp(logvar_p, -10, 10)

                var_q = torch.exp(logvar_q)
                var_p = torch.exp(logvar_p)
                kl_fine = 0.5 * torch.sum(
                    (var_q / var_p) + ((mu_p - mu_q)**2) / var_p 
                    - 1 + (logvar_p - logvar_q),
                    dim=[1,2,3]
                ).mean()
                
            # Loss Calculation Logic
            
            # 1. VQ-VAE Path
            if args.model_type == 'vqvae':
                # VQ-VAE: Loss = Recon + VQ(Commitment). Prior interaction is future work.
                # vq_loss is already calculated in forward pass
                recon_loss = calc_reconstruction_mse(x_rec, x)
                total_loss = recon_loss + vq_loss
                
                prior_loss = vq_loss # For logging
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                vae_optimizer.step()
                
            # 2. Standard VAE Path (VAE, NVAE)
            else:
                recon_loss = calc_reconstruction_mse(x_rec, x)
                prior_loss = torch.tensor(0.0, device=device)
                
                # Coarse/Global Prior Loss
                if args.prior == 'kl':
                    # Clamp coarse logvar too if needed
                    if args.model_type == 'nvae':
                         logvar = torch.clamp(logvar, -10, 10)
                    p_loss, _ = prior_criterion(mean, logvar)
                    prior_loss = p_loss
                       
                elif args.prior == 'mmd':
                    if args.model_type == 'vae':
                        p_loss, _ = prior_criterion(z_for_prior)
                    else: # NVAE coarse latent [B, D]
                        p_loss, _ = prior_criterion(z_for_prior.flatten(1)) 
                    prior_loss = p_loss
                    
                elif args.prior == 'imm':
                    mmd_reg_fn = MMDLoss(mmd_weight=args.mmd_weight)
                    mmd_loss, _ = mmd_reg_fn(z_for_prior.flatten(1))
                    
                    imm_loss_val = prior_net.compute_loss(z_for_prior.flatten(1).detach())
                    prior_loss = imm_loss_val
                    recon_loss = recon_loss + mmd_loss
                    
                elif args.prior == 'ddpm':
                    mmd_reg_fn = MMDLoss(mmd_weight=args.mmd_weight)
                    mmd_loss, _ = mmd_reg_fn(z_for_prior.flatten(1))
                    
                    ddpm_loss_val, _ = ddpm_criterion(prior_net, z_for_prior.flatten(1).detach(), device=device)
                    prior_loss = ddpm_loss_val
                    recon_loss = recon_loss + mmd_loss
    
                # Backward Pass
                if args.prior in ['kl', 'mmd']:
                    total_loss = recon_loss + prior_loss
                    if args.model_type == 'nvae':
                        total_loss = total_loss + kl_fine * args.kl_weight 
                        
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                    if prior_net:
                         total_norm = torch.nn.utils.clip_grad_norm_(prior_net.parameters(), 1.0)
                         if step % args.log_interval == 0:
                             print(f"   [Grad Norm] Prior: {total_norm:.4f}")
    
                    vae_optimizer.step()
                else:
                    # Two-stage-like update
                    vae_loss = recon_loss
                    if args.model_type == 'nvae':
                        vae_loss += kl_fine * args.kl_weight
                        
                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                    vae_optimizer.step()
                    
                    prior_loss.backward()
                    if prior_net:
                         total_norm = torch.nn.utils.clip_grad_norm_(prior_net.parameters(), 1.0)
                         if step % args.log_interval == 0:
                             print(f"   [Grad Norm] Prior: {total_norm:.4f}")
                    prior_optimizer.step()
                
            if step % args.log_interval == 0:
                log_str = f"Epoch {epoch} | Step {step} | Recon: {recon_loss.item():.4f} | Prior: {prior_loss.item():.4f}"
                if args.model_type == 'nvae':
                    log_str += f" | FineKL: {kl_fine.item():.4f}"
                print(log_str)
        
        ckpt_path = os.path.join(args.save_dir, f"ckpt_{args.prior}_epoch{epoch}.pth")
        torch.save({'vae': vae.state_dict(), 'prior': prior_net.state_dict() if prior_net else None}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, required=True, choices=['kl', 'mmd', 'imm', 'ddpm'], help='Type of prior to train')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr_vae', type=float, default=2e-4, help='VAE Learning Rate')
    parser.add_argument('--lr_prior', type=float, default=1e-4, help='Prior Network Learning Rate')
    parser.add_argument('--kl_weight', type=float, default=0.00025, help='Weight for KL loss')
    parser.add_argument('--mmd_weight', type=float, default=10.0, help='Weight for MMD loss')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval steps')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset')
    
    args = parser.parse_args()
    
    # 1. Dataset Setup
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Data nesting: args.data_root/CelebA
        # Note: torchvision CelebA expects 'root' to contain the zip or 'celeba' folder.
        # If user has e:/imm_vae/data/CelebA, passing that as root should work.
        dset_root = os.path.join(args.data_root, 'CelebA')
        print(f"Loading CelebA from {dset_root}...")
        try:
            # os.makedirs(dset_root, exist_ok=True) # Don't make if we expect it to exist
            train_dataset = datasets.CelebA(root=dset_root, split='train', download=False, transform=transform)
            val_dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
        except Exception as e:
            print(f"Could not load CelebA from {dset_root}: {e}. Aborting.")
            exit(1)
            
    elif args.dataset == 'CIFAR10':
        t_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.model_type == 'vae':
            t_list.insert(0, transforms.Resize(64))
        
        transform = transforms.Compose(t_list)
        
        dset_root = os.path.join(args.data_root, 'CIFAR10')
        print(f"Loading CIFAR10 from {dset_root}...")
        try:
            # os.makedirs(dset_root, exist_ok=True)
            train_dataset = datasets.CIFAR10(root=dset_root, train=True, download=False, transform=transform)
            val_dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
        except Exception as e:
             print(f"Could not load CIFAR10 from {dset_root}: {e}. Aborting.")
             exit(1)

    elif args.dataset == 'FashionMNIST':
        # 1-channel, 28x28. Resize to 32x32.
        t_list = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform = transforms.Compose(t_list)
        
        dset_root = os.path.join(args.data_root, 'FashionMNIST')
        print(f"Loading FashionMNIST from {dset_root}...")
        try:
            train_dataset = datasets.FashionMNIST(root=dset_root, train=True, download=True, transform=transform)
            val_dataset = datasets.FashionMNIST(root=dset_root, train=False, download=True, transform=transform)
        except Exception as e:
             print(f"Could not load FashionMNIST: {e}. Aborting.")
             exit(1)

    elif args.dataset == 'FFHQ':
        # FFHQ (128x128)
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dset_root = os.path.join(args.data_root, 'FFHQ')
        print(f"Loading FFHQ from {dset_root}...")
        try:
             full_dataset = FlatImageFolder(root=dset_root, transform=transform)
             # 90/10 Split
             total = len(full_dataset)
             train_len = int(total * 0.9)
             val_len = total - train_len
             train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])
        except Exception as e:
             print(f"Could not load FFHQ from {dset_root}: {e}. Aborting.")
             exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)

    train(args, train_loader, val_loader)
