
import time
import os
import subprocess
import sys

def run_evaluation(priority, ckpt_path):
    print(f"[Watcher] Found completed checkpoint for {priority}: {ckpt_path}")
    print(f"[Watcher] Starting Evaluation Suite for {priority}...")
    
    # Define parameters based on prior
    steps = "50"
    if priority == 'ddpm': steps = "1000"
    
    python_exe = sys.executable
    cmd = [
        python_exe, "e:/imm_vae/imm-main/run_evaluation_suite.py",
        "--ckpt_path", ckpt_path,
        "--prior", priority,
        "--dataset", "CelebA",
        "--model_type", "vae",
        "--data_root", "e:/imm_vae/data",
        "--num_samples", "2000",
        "--steps", steps
    ]
    
    log_file = os.path.join(os.path.dirname(ckpt_path), f"eval_suite_{priority}.log")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"[Watcher] Evaluation for {priority} completed. Log: {log_file}")

def main():
    base_root = "e:/imm_vae/checkpoints_v2"
    datasets = ['CelebA', 'CIFAR10', 'FashionMNIST']
    model_types = ['vae', 'nvae', 'vqvae']
    priors = ['kl', 'mmd', 'imm', 'ddpm']
    
    # Track status per combo: status[dataset][model][prior] = True/False
    status = {}
    for d in datasets:
        status[d] = {}
        for m in model_types:
            status[d][m] = {p: False for p in priors}
            
    print(f"[Watcher] Monitoring {base_root} for all experiments...")
    
    while True:
        for dataset in datasets:
            for model_type in model_types:
                base_dir = os.path.join(base_root, dataset, model_type)
                
                if not os.path.exists(base_dir):
                    continue
                    
                for prior in priors:
                    if status[dataset][model_type][prior]:
                        continue
                        
                    ckpt_name = f"ckpt_{prior}_epoch49.pth"
                    ckpt_path = os.path.join(base_dir, ckpt_name)
                    
                    if os.path.exists(ckpt_path):
                        time.sleep(10) # Wait for write
                        
                        marker_path = os.path.join(base_dir, f"eval_done_{prior}.txt")
                        if not os.path.exists(marker_path):
                            print(f"[Watcher] Detected {dataset}/{model_type}/{prior}")
                            try:
                                # Call run_evaluation with correct args
                                steps = "50"
                                if prior == 'ddpm': steps = "1000"
                                
                                python_exe = sys.executable
                                cmd = [
                                    python_exe, "e:/imm_vae/imm-main/run_evaluation_suite.py",
                                    "--ckpt_path", ckpt_path,
                                    "--prior", prior,
                                    "--dataset", dataset,
                                    "--model_type", model_type,
                                    "--data_root", "e:/imm_vae/data",
                                    "--num_samples", "2000",
                                    "--steps", steps
                                ]
                                log_file = os.path.join(base_dir, f"eval_suite_{prior}.log")
                                
                                with open(log_file, "w") as f:
                                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                                    
                                with open(marker_path, "w") as f:
                                    f.write("Done")
                                    
                                print(f"[Watcher] Finished {dataset}/{model_type}/{prior}")
                                status[dataset][model_type][prior] = True
                            except Exception as e:
                                print(f"[Watcher] Error: {e}")
                        else:
                            status[dataset][model_type][prior] = True
        
        time.sleep(60)

if __name__ == "__main__":
    main()
