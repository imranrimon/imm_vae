
import subprocess
import sys
import os
import time

# Configuration
python_exe = sys.executable
python_cmd = [python_exe, "-u"]
dataset = 'CelebA'
model_type = 'vae'
data_root = "e:/imm_vae/data"
save_dir = f"e:/imm_vae/checkpoints_v2/{dataset}/{model_type}"
epochs = 50
z_dim = 128

wait_pid = 38976

def is_process_running(pid):
    # Cross-platform check
    if os.name == 'nt':
        # Windwos
        try:
            # check if PID exists in tasklist
            output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode()
            return str(pid) in output
        except:
            return False
    else:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

def run_command(cmd_list, log_file=None):
    print(f"Running: {' '.join(cmd_list)}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n\n--- Running: {' '.join(cmd_list)} ---\n")
            subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd_list, check=True)

def main():
    log_file = os.path.join(save_dir, "experiment.log")
    
    print(f"Waiting for MMD process {wait_pid} to finish...")
    while is_process_running(wait_pid):
        time.sleep(30)
        print(f"Still waiting for PID {wait_pid}...")
        
    print("MMD finished (or process died). Proceeding to IMM and DDPM.")
    
    priors_remaining = ['imm', 'ddpm']
    
    for prior in priors_remaining:
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
