
import argparse
import torch
import os
from torchvision.utils import save_image, make_grid
from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper
from training.imm_utils import IMM, SimpleDDPM, OtFmEulerFmParameterization, ResidualDenoiser
from training.sampling_utils import sample_kl_mmd, sample_imm, sample_ddpm

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing Samples: {args.dataset} | {args.model_type} | {args.prior}")

    # 1. Model Setup
    input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
    if args.dataset == 'FFHQ': input_res = 128
    
    in_channels = 1 if args.dataset == 'FashionMNIST' else 3
    z_dim = args.z_dim # Use argument

    if args.model_type == 'vae':
        vae = Model(z_dim=z_dim, in_channels=in_channels, input_res=input_res).to(device)
    elif args.model_type == 'nvae':
        vae = NVAE(z_dim=z_dim, input_res=input_res).to(device)
    elif args.model_type == 'vqvae':
        # VQ-VAE usualy samples by training a prior (PixelCNN/Transformer) on indices.
        # But if prior='kl' (standard VQVAE training?), we might not have a sampler?
        # Our implementation treats VQ-VAE as autoencoder. 
        # If user provided a 'prior' model (not supported in simple VQVAE setup yet), we'd use it.
        # However, our VQVAE wrapper exposes decoder(z). We can sample continuous z and verify?
        # Or more likely, VQVAE sampling requires the 'prior_net' if trained?
        # Given current setup, VQ-VAE is just reconstruction/commitment. 
        # Sampling naive normal z might fail if quantization is strict.
        # We will try robust sampling or skip.
        vae = VQVAE_Wrapper(z_dim=z_dim, input_res=input_res, in_channels=in_channels).to(device)

    # Prior Setup
    prior_net = None
    if args.prior == 'imm':
         param = OtFmEulerFmParameterization(sigma_data=1.0)
         denoiser = ResidualDenoiser(input_dim=z_dim + 2, model_dim=512, n_blocks=6, output_dim=z_dim)
         prior_net = IMM(network=denoiser, parameterization=param, D=z_dim).to(device)
    elif args.prior == 'ddpm':
         denoiser = ResidualDenoiser(input_dim=z_dim + 1, model_dim=512, n_blocks=6, output_dim=z_dim)
         prior_net = SimpleDDPM(network=denoiser, z_dim=z_dim).to(device)
         
    # Load Checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    vae.load_state_dict(checkpoint['vae'])
    if prior_net and 'prior' in checkpoint and checkpoint['prior'] is not None:
        prior_net.load_state_dict(checkpoint['prior'])
        
    vae.eval()
    if prior_net: prior_net.eval()

    # 2. Sample
    torch.manual_seed(42) # Fixed seed for consistency across experiments
    
    with torch.no_grad():
        if args.prior in ['kl', 'mmd']:
            samples = sample_kl_mmd(vae, args.n_imgs, device)
        elif args.prior == 'imm':
            samples = sample_imm(vae, prior_net, args.n_imgs, device, steps=args.steps)
        elif args.prior == 'ddpm':
            samples = sample_ddpm(vae, prior_net, args.n_imgs, device, steps=args.steps)
            
    # 3. Save
    samples = (samples + 1) / 2
    grid = make_grid(samples, nrow=8)
    
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"samples_{args.dataset}_{args.model_type}_{args.prior}.png")
    save_image(grid, save_path)
    print(f"Saved samples to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--prior', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--n_imgs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=50) # For IMM
    parser.add_argument('--z_dim', type=int, default=128)
    args = parser.parse_args()
    main(args)
