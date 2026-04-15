
import subprocess
import sys
import os
import torch

# Configuration
python_exe = sys.executable
# Ensure unbuffered output for real-time logging
python_cmd = [python_exe, "-u"]
datasets = ['CelebA', 'CIFAR10', 'FashionMNIST']
model_types = ['vae', 'nvae', 'vqvae']
priors = ['kl', 'mmd', 'imm', 'ddpm'] # Run all 4
epochs = 50 # Production run
z_dim = 128
data_root = "e:/imm_vae/data" # Ensure this path is correct for user environment

# Base checkpoint dir
base_ckpt_dir = "e:/imm_vae/checkpoints_v2" # New dir for V2 experiments

def run_command(cmd_list, log_file=None):
    print(f"Running: {' '.join(cmd_list)}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n\n--- Running: {' '.join(cmd_list)} ---\n")
            subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd_list, check=True)

def main():
    os.makedirs(base_ckpt_dir, exist_ok=True)
    
    for dataset in datasets:
        for model_type in model_types:
            print(f"==================================================")
            print(f"Starting Experiment: Dataset={dataset}, Model={model_type}")
            print(f"==================================================")
            
            save_dir = os.path.join(base_ckpt_dir, dataset, model_type)
            os.makedirs(save_dir, exist_ok=True)
            log_file = os.path.join(save_dir, "experiment.log")
            
            # 1. Training & Evaluation Loop
            # For VQ-VAE, prior argument is ignored (loss is just vq_loss). Avoid redundant runs.
            current_priors = ['kl'] if model_type == 'vqvae' else priors
            
            for prior in current_priors:
                print(f"--- Processing {prior.upper()} ---")
                ckpt_path = os.path.join(save_dir, f"ckpt_{prior}_epoch{epochs-1}.pth")
                
                # A. Training
                if os.path.exists(ckpt_path):
                     print(f"Checkpoint {ckpt_path} exists. Skipping training.")
                else:
                    print(f"--- Training {prior.upper()} ---")
                    cmd = [
                        *python_cmd, "e:/imm_vae/imm-main/train_joint.py",
                        "--prior", prior,
                        "--dataset", dataset,
                        "--model_type", model_type,
                        "--z_dim", str(z_dim),
                        "--epochs", str(epochs),
                        "--save_dir", save_dir,
                        "--data_root", data_root,
                        "--log_interval", "200",
                        "--mmd_weight", "1.0"
                    ]
                    if prior == 'imm':
                        cmd.extend(["--lr_prior", "1e-5"])
                    try:
                        run_command(cmd, log_file)
                    except subprocess.CalledProcessError as e:
                        print(f"Training failed for {prior}: {e}")
                        continue

                # B. Evaluation
                if not os.path.exists(ckpt_path):
                    print(f"Checkpoint {ckpt_path} not found. Skipping evaluation.")
                    continue
                
                print(f"--- Evaluating {prior.upper()} ---")
                try:
                    # Use the unified evaluation suite
                    # Determine steps 
                    steps = "50"
                    if prior == 'ddpm': steps = "1000"
                    
                    eval_cmd = [
                        *python_cmd, "e:/imm_vae/imm-main/run_evaluation_suite.py",
                        "--ckpt_path", ckpt_path,
                        "--prior", prior,
                        "--dataset", dataset,
                        "--model_type", model_type,
                        "--data_root", data_root,
                        "--num_samples", "2000",
                        "--steps", steps
                    ]
                    run_command(eval_cmd, log_file)
                except Exception as e:
                    print(f"Evaluation suite failed for {prior}: {e}")

if __name__ == "__main__":
    main()
