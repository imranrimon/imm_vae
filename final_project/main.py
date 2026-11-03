
import argparse
import torch
from src.training.stage1 import train_stage1
from src.training.stage2 import train_stage2
# from src.evaluation.metrics import evaluate_suite # TODO

def main():
    parser = argparse.ArgumentParser(description="IMM/VAE Final Project CLI")
    
    # Core Arguments
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help="Stage 1: Base, Stage 2: Prior, Stage 3: Eval")
    parser.add_argument('--model', type=str, choices=['vae', 'nvae'], default='vae', help="Base Model Type")
    parser.add_argument('--dataset', type=str, choices=['CelebA', 'CIFAR10', 'FFHQ'], default='CIFAR10', help="Dataset")
    parser.add_argument('--method', type=str,
                        choices=['kl', 'mmd', 'ddpm', 'imm', 'meanflow', 'meanflow_improved'],
                        help="Method (Stage 1: kl/mmd, Stage 2: ddpm/imm/meanflow/meanflow_improved)")
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--z_dim', type=int, default=64, help="Latent dimension")
    parser.add_argument('--kl_weight', type=float, default=None, help="KL Weight (overrides default 0.00025)")
    parser.add_argument('--base_method', type=str, default='kl', choices=['kl', 'mmd'], help="Base method for Stage 2/3 (kl or mmd)")
    parser.add_argument('--lr', type=float, default=None, help="Stage 2 learning rate (default: per-method)")
    parser.add_argument('--jvp_free', action='store_true', help="MeanFlow: finite-difference instead of exact JVP")
    parser.add_argument('--z_fine_ch', type=int, default=None, help="NVAE fine-latent channels (default 128 = original over-complete code)")
    parser.add_argument('--free_bits', type=float, default=0.0, help="Free bits (nats) per latent dim")
    parser.add_argument('--run_tag', type=str, default='',
                        help="Suffix distinguishing runs that share a name "
                             "(e.g. a data-preprocessing change)")
    parser.add_argument('--kl_warmup', type=int, default=0, help="Epochs to linearly ramp the KL weight")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    if args.stage == 1:
        if args.method not in ['kl', 'mmd']:
            print("Error: For Stage 1, method must be 'kl' or 'mmd'")
            return
        train_stage1(args, device)
        
    elif args.stage == 2:
        if args.method not in ['ddpm', 'imm', 'meanflow', 'meanflow_improved']:
            print("Error: For Stage 2, method must be one of "
                  "'ddpm', 'imm', 'meanflow', 'meanflow_improved'")
            return
        train_stage2(args, device)
        
    elif args.stage == 3:
        print("Stage 3: Evaluation Suite")
        # evaluate_suite(args, device)
        print("Please run 'Research_Suite.ipynb' for interactive visualization and analysis.")
        
if __name__ == "__main__":
    main()
