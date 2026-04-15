
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from training.simple_ae import Model
from training.imm_utils import ResidualDenoiser, SimpleDDPM
from training.loss import DDPMLoss, MMDLoss

def calc_reconstruction_mse(x_recon, x):
    return torch.nn.functional.mse_loss(x_recon, x, reduction='mean')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DDPM on {device} | Dataset: {args.dataset}")

    # 1. Dataset
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dset_root = os.path.join(args.data_root, 'CelebA')
        train_dataset = datasets.CelebA(root=dset_root, split='train', download=False, transform=transform)
        val_dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported in this script yet.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)

    # 2. VAE Model
    input_res = 64
    vae = Model(z_dim=args.z_dim, in_channels=3, input_res=input_res).to(device)
    
    # 3. DDPM Prior
    # Network expects input [B, z_dim + time_dim]
    denoiser = ResidualDenoiser(input_dim=args.z_dim + 128, model_dim=512, n_blocks=6, output_dim=args.z_dim)
    prior_net = SimpleDDPM(network=denoiser, z_dim=args.z_dim, time_dim=128).to(device)
    
    optimizer_vae = optim.Adam(vae.parameters(), lr=2e-4)
    optimizer_prior = optim.Adam(prior_net.parameters(), lr=1e-4)
    
    ddpm_criterion = DDPMLoss(noise_schedule="cosine")
    mmd_criterion = MMDLoss(mmd_weight=args.mmd_weight)

    # Resume logic
    start_epoch = 0
    if os.path.exists(args.save_dir):
        # Find latest
        pass # Implement as needed

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        vae.train()
        prior_net.train()
        
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            
            # Forward VAE
            z, x_rec, mean, logvar = vae(x)
            z_flat = z.flatten(1)
            
            # 1. VAE Loss (Recon + MMD)
            recon_loss = calc_reconstruction_mse(x_rec, x)
            mmd_loss, _ = mmd_criterion(z_flat)
            
            # 2. DDPM Loss (Prior)
            # Train DDPM to denoising the *detached* latent
            # We detach because we don't want DDPM loss to shape the latent space directly involved in reconstruction(?).
            # Usually Latent Diffusion trains on fixed latents.
            # But here we train jointly.
            # If we train jointly, maybe we should differentiate through z?
            # Standard LDM trains on pre-trained VAE.
            # Here we follow train_joint.py pattern: detach z for DDPM?
            # train_joint.py used: ddpm_loss_val, _ = ddpm_criterion(prior_net, z_for_prior.flatten(1).detach(), ...)
            # So yes, detached.
            
            ddpm_loss, _ = ddpm_criterion(prior_net, z_flat.detach(), device=device)
            
            total_loss_vae = recon_loss + mmd_loss
            total_loss_prior = ddpm_loss
            
            # Optimization
            optimizer_vae.zero_grad()
            total_loss_vae.backward()
            optimizer_vae.step()
            
            optimizer_prior.zero_grad()
            total_loss_prior.backward()
            optimizer_prior.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] Recon: {recon_loss.item():.4f} MMD: {mmd_loss.item():.4f} DDPM: {ddpm_loss.item():.4f}")
        
        # Save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"ckpt_ddpm_epoch{epoch}.pth")
        torch.save({'vae': vae.state_dict(), 'prior': prior_net.state_dict()}, ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--data_root', type=str, default='e:/imm_vae/data')
    parser.add_argument('--save_dir', type=str, default='e:/imm_vae/checkpoints_new_ddpm')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--mmd_weight', type=float, default=10.0) # From train_joint default
    args = parser.parse_args()
    main(args)
