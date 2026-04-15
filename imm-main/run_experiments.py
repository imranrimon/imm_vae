
import subprocess
import os
import argparse
import sys

# Configuration
DATASETS = ['CelebA', 'CIFAR10', 'FFHQ']
MODELS = {
    'CelebA': 'vae',   # or nvae, vqvae
    'CIFAR10': 'vae',
    'FFHQ': 'vae'
}
PRIORS = ['kl', 'mmd'] # Evaluating both KL and MMD
Z_DIM = 128
BATCH_SIZES = {
    'CelebA': 64,
    'CIFAR10': 64,
    'FFHQ': 32 # 128x128 
}
EPOCHS = 50 # Default for demo/quick run. User can increase.

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error running command: {cmd}")
        # sys.exit(1) # Continue or exit? Let's continue to next if one fails?
        # But training failure is critical.
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_train', action='store_true', help='Skip training, run eval only')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    python_exe = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_joint.py")
    eval_script = os.path.join(script_dir, "run_evaluation_suite.py")
    
    results_dir = os.path.join(script_dir, "..", "checkpoints_v2") # separate dir for these runs
    
    for dataset in DATASETS:
        print(f"\n{'='*40}\nProcessing {dataset}\n{'='*40}")
        
        model_type = MODELS[dataset]
        batch_size = BATCH_SIZES[dataset]
        
        for prior in PRIORS:
            # Checkpoint Name
            # format: {dataset}/{model_type}/{prior}/model.pt
            ckpt_dir = os.path.join(results_dir, dataset, model_type, prior)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{prior}_epoch{args.epochs-1}.pth")
            
            # 1. Training
            if not args.skip_train:
                print(f"--- Training {dataset} ({prior}) ---")
                train_cmd = f'{python_exe} "{train_script}" --dataset {dataset} --model_type {model_type} --prior {prior} --z_dim {Z_DIM} --batch_size {batch_size} --epochs {args.epochs} --save_dir "{ckpt_dir}" --data_root "e:/imm_vae/data"'
                
                # Check if exists (optional skip logic if desired, but --skip_train handles it)
                if os.path.exists(ckpt_path):
                     print(f"Checkpoint already exists at {ckpt_path}. Overwriting/Resuming based on script logic (usually overwrite if simple run).")
                     
                run_command(train_cmd)
            
            # 2. Evaluation
            print(f"--- Evaluating {dataset} ---")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}. Skipping evaluation.")
                continue
                
            eval_cmd = f'{python_exe} "{eval_script}" --ckpt_path "{ckpt_path}" --dataset {dataset} --model_type {model_type} --prior {prior} --data_root "e:/imm_vae/data"'
            run_command(eval_cmd)
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
