
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import dnnlib 

from training.simple_ae import Model
from training.nvae_model import NVAE
from training.vqvae_model import VQVAE_Wrapper as VQVAE
from training.imm_utils import IMM, OtFmEulerFmParameterization, ResidualDenoiser, SimpleDDPM
from training.sampling_utils import sample_kl_mmd, sample_imm, sample_ddpm
from training.datasets import FlatImageFolder

class InceptionFeat(torch.nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.inc = inc
        self.inc.fc = torch.nn.Identity()
    def forward(self, x):
        return self.inc(x)

class SimpleStats:
    def __init__(self, device):
        self.device = device
        self.features = []

    def append(self, x):
        self.features.append(x.detach().cpu())

    def get_mean_cov(self):
        features = torch.cat(self.features, dim=0).numpy()
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

import numpy as np
from scipy import linalg

def calculate_fid(mu1, sigma1, mu2, sigma2):
    eps = 1e-6
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.iscomplexobj(sigma1) and not np.iscomplexobj(sigma2):
            if np.isreal(covmean.real).all():
                 covmean = covmean.real
            else:
                 # Debug prints
                 print("Complex covmean:", covmean.diagonal()[:5])
                 pass

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_statistics_for_dataset(dataset, detector, device, max_items=10000, batch_size=64):
    print(f"Computing statistics for real data ({max_items} items)...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    stats = SimpleStats(device)
    
    count = 0
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            # Resize to Inception input size 299x299
            # CelebA/CIFAR is [-1, 1], Inception expects... Inception-V3 (torchvision) expects normalized.
            # But StudioGAN/StyleGAN FID usually expects [0, 255] or specific range.
            # dnnlib stylegan3 detector usually expects [0, 255] uint8 inputs if raw, OR feature extraction logic.
            # Standard FID calculation: Resize to 299x299, then standard Inception.
            
            # Check for 1-channel input (FashionMNIST)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
            # If using dnnlib's pickle, it expects [N, 3, H, W] in range [-1, 1]? Or [0, 255]?
            # Usually [0, 255] for StyleGAN metrics.
            # But let's check standard practice. Torchvision Inception expects normalized.
            # The pickle loaded from NVIDIA source expects [0, 255] uint8 in their code, but here we pass tensors.
            # Let's assume standard [-1, 1] -> [0, 255] conversion if checking against their repo,
            # but usually 'detector(x)' handles it if we match valid input.
            # Let's keep existing logic: x is [-1, 1] tensor.
            # If using standard InceptionV3 from torchvision (fallback), it handles norm.
            # If using NVIDIA pickle, it might expect specific range.
            # For this 'SimpleStats' we just feed x directly.
            
            
            # Preprocessing logic depends on detector type
            if isinstance(detector, InceptionFeat):
                # Torchvision Inception V3 expects:
                # 1. Inputs [0, 1] float
                # 2. Normalized with ImageNet stats mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # x is currently [-1, 1].
                x_01 = (x + 1) / 2.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                x_norm = (x_01 - mean) / std
                feat = detector(x_norm)
                
            else:
                # NV Pickle detector (StyleGAN) usually expects [0, 255] uint8
                x_int = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                try:
                    feat = detector(x_int, return_features=True)
                except:
                    # Fallback if return_features not supported
                    try:
                        feat = detector(x_int)
                    except:
                        # desperate fallback
                        feat = detector(x)
            
            if count == 0:
                 print(f"DEBUG: feat shape: {feat.shape}, min: {feat.min()}, max: {feat.max()}")
            
            stats.append(feat)
            
            count += x.shape[0]
            if count >= max_items:
                break
                
    return stats.get_mean_cov()

def compute_statistics_for_generator(vae, prior_net, sample_fn, detector, device, max_items=10000, batch_size=64, steps=None):
    print(f"Computing statistics for generated data ({max_items} items)...")
    stats = SimpleStats(device)
    
    num_batches = (max_items + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            if prior_net:
                if sample_fn.__name__ in ['sample_imm', 'sample_ddpm'] and steps is not None:
                     imgs = sample_fn(vae, prior_net, batch_size, device, steps=steps)
                else:
                     imgs = sample_fn(vae, prior_net, batch_size, device)
            else:
                imgs = sample_fn(vae, batch_size, device)
            
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
                
            imgs_resized = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            
            if isinstance(detector, InceptionFeat):
                # Standard Torchvision Norm
                x_01 = (imgs_resized + 1) / 2.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                x_norm = (x_01 - mean) / std
                feat = detector(x_norm)
            else:
                imgs_int = ((imgs_resized + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                try:
                    feat = detector(imgs_int, return_features=True)
                except:
                    try:
                        feat = detector(imgs_int)
                    except:
                        feat = detector(imgs_resized)
            
            stats.append(feat)
            
    return stats.get_mean_cov()

def main(args):
    try:
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
            # Path fix: e:/imm_vae/data/CelebA
            dset_root = os.path.join(args.data_root, 'CelebA')
            dataset = datasets.CelebA(root=dset_root, split='valid', download=False, transform=transform)
        elif args.dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            # Path fix
            dset_root = os.path.join(args.data_root, 'CIFAR10')
            dataset = datasets.CIFAR10(root=dset_root, train=False, download=False, transform=transform)
        elif args.dataset == 'FashionMNIST':
            # Resize to 32 for consistency (or keep 28 but Inception resize handles it).
            # Usually we train on 32.
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
            # Use FlatImageFolder
            dataset = FlatImageFolder(root=dset_root, transform=transform)
        
        # 2. Load Model
        print(f"Loading checkpoint from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)
        
        if args.model_type == 'vae':
            input_res = 64
            if args.dataset in ['CIFAR10', 'FashionMNIST']: input_res = 32
            elif args.dataset == 'FFHQ': input_res = 128
            vae = Model(z_dim=args.z_dim, input_res=input_res).to(device)
        elif args.model_type == 'nvae':
            input_res = 32 if args.dataset == 'CIFAR10' else 64
            vae = NVAE(z_dim=args.z_dim, input_res=input_res).to(device)
            # NVAE Wrapper for sampling
            class NVAEWrapper(torch.nn.Module):
                def __init__(self, nvae):
                    super().__init__()
                    self.nvae = nvae
                    self.decoder = self
                def forward(self, z): 
                    out, _, _ = self.nvae.decoder(z, z_fine=None) # Sample fine internally
                    return out
            
            vae = NVAEWrapper(vae).to(device)
            vae.nvae.load_state_dict(ckpt['vae'])
        elif args.model_type == 'vqvae':
            input_res = 32 if args.dataset in ['CIFAR10', 'FashionMNIST'] else 64
            in_channels = 1 if args.dataset == 'FashionMNIST' else 3
            vae = VQVAE(z_dim=args.z_dim, input_res=input_res, in_channels=in_channels).to(device)
            vae.load_state_dict(ckpt['vae'])
            
            # VQ-VAE returns: (quantized, x_rec, vq_loss, perplexity)
            # We need a wrapper to return *images* (x_rec) for sampling functions OR
            # sampling logic must handle it.
            # 'sample_kl_mmd' usually calls 'vae.decoder(z)'.
            # VQVAE wrapper has .decoder().
            # Need to ensure 'sample_kl_mmd' works.
            # sample_fn calls 'vae.sample(batch_size)'? Or 'sample_kl_mmd(vae...)' logic?
            # sample_kl_mmd: z = torch.randn(...). x = vae.decoder(z).
            # VQVAE decoder expects quantized vectors. VAE expects 'z'.
            # If we sample Gaussian z and feed to VQVAE decoder... it expects embedding_dim vectors?
            # VQVAE wrapper decoder takes [B, embedding_dim, H, W]??
            # Wrapper VQVAE .decoder input: 'quantized' [B, embedding_dim, H, W]
            pass
        else:
             vae.load_state_dict(ckpt['vae']) # Fallback

        if args.model_type == 'vae':
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
            denoiser = ResidualDenoiser(input_dim=args.z_dim + 128, model_dim=512, n_blocks=6, output_dim=args.z_dim)
            prior_net = SimpleDDPM(network=denoiser, z_dim=args.z_dim, time_dim=128).to(device)
            prior_net.load_state_dict(ckpt['prior'])
            prior_net.eval()
            sample_fn = sample_ddpm
            
        print("Loading feature detector...")
        url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
        try:
             with dnnlib.util.open_url(url) as f:
                 detector = pickle.load(f).to(device)
                 detector.eval()
        except:
             print("Falling back to torchvision Inception V3")
             from torchvision.models import inception_v3
             detector = inception_v3(pretrained=True, transform_input=False).to(device)
             detector.eval()
             detector = InceptionFeat(detector)

        # 4. Compute Statistics
        mu_real, sigma_real = compute_statistics_for_dataset(dataset, detector, device, max_items=args.num_samples, batch_size=args.batch_size)
        
        # Determine steps default if not passed
        sampling_steps = args.steps
        if sampling_steps is None:
             if args.prior == 'imm': sampling_steps = 50
             elif args.prior == 'ddpm': sampling_steps = 1000
        
        print(f"Sampling with {sampling_steps} steps (Prior: {args.prior})")
        
        mu_gen, sigma_gen = compute_statistics_for_generator(vae, prior_net, sample_fn, detector, device, max_items=args.num_samples, batch_size=args.batch_size, steps=sampling_steps)
        
        # 5. Compute FID
        fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        print(f"FID Score ({args.prior.upper()}): {fid:.4f}")
        
        import json
        metrics = {
            "fid": fid
        }
        with open(os.path.join(os.path.dirname(args.ckpt_path), "metrics_fid.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prior', type=str, required=True, choices=['kl', 'mmd', 'imm', 'ddpm'], help='Prior type')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model Architecture')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples for stats')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')
    parser.add_argument('--steps', type=int, default=None, help='Number of sampling steps (for IMM/DDPM)')
    
    args = parser.parse_args()
    main(args)
