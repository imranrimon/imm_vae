
import subprocess
import sys
import os

# Configuration
python_exe = sys.executable
# Standard python command. No "-u" needed for interactive terminal usually, but safe to keep.
python_cmd = [python_exe]

# Experiment Settings
dataset = 'CelebA'
model_type = 'vae'
data_root = "e:/imm_vae/data"
save_dir = f"e:/imm_vae/checkpoints_v2/{dataset}/{model_type}"
epochs = 50
z_dim = 128

# Sequence: MMD -> IMM -> DDPM (KL is already done, so skipping unless requested)
# User asked: Resume MMD -> Eval -> Resume IMM -> Eval -> DDPM -> Eval
priors_sequence = ['mmd', 'imm', 'ddpm']

def run_interactive(cmd_list):
    print(f"\n\n>>> Running: {' '.join(cmd_list)}")
    print(">>> Output will appear below (Progress Bar should be visible)...\n")
    # shell=False to ensure signals propagate, check=True to stop on error
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Error running command. Exit Code: {e.returncode}")
        # Ask user if they want to continue
        input("Press Enter to continue to next step (or Ctrl+C to stop)...")

def main():
    print(f"Starting Interactive Run for {dataset} / {model_type}")
    print(f"Sequence: {priors_sequence}")
    print("Note: train_joint.py has been updated to RESUME from checkpoints automatically.")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for prior in priors_sequence:
        print(f"\n\n==================================================")
        print(f"   PHASE: {prior.upper()}")
        print(f"==================================================")
        
        # 1. Training (Resumes automatically)
        print(f"--- Step 1: Training {prior.upper()} ---")
        train_cmd = [
            *python_cmd, "e:/imm_vae/imm-main/train_joint.py",
            "--prior", prior,
            "--dataset", dataset,
            "--model_type", model_type,
            "--z_dim", str(z_dim),
            "--epochs", str(epochs),
            "--save_dir", save_dir,
            "--data_root", data_root,
            "--log_interval", "200", # Logging step
            "--mmd_weight", "1.0"
        ]
        if prior == 'imm':
            train_cmd.extend(["--lr_prior", "1e-5"])
            
        run_interactive(train_cmd)
        
        # 2. Evaluation
        print(f"--- Step 2: Evaluating {prior.upper()} ---")
        
        # Check if checkpoint exists
        ckpt_path = os.path.join(save_dir, f"ckpt_{prior}_epoch{epochs-1}.pth")
        
        # Wait for file to write? (Usually immediate after process exit)
        if not os.path.exists(ckpt_path):
             print(f"Warning: Checkpoint {ckpt_path} not found. Trying to find ANY checkpoint...")
             # Fallback to latest
             existing = [f for f in os.listdir(save_dir) if f.startswith(f"ckpt_{prior}_epoch")]
             if existing:
                 # Sort by epoch
                 existing.sort(key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))
                 ckpt_path = os.path.join(save_dir, existing[-1])
                 print(f"Found alternative: {ckpt_path}")
             else:
                 print("No checkpoint found. Skipping evaluation.")
                 continue
                 
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
        run_interactive(eval_cmd)

    print("\n\n>>> ALL DONE! <<<")

if __name__ == "__main__":
    main()
