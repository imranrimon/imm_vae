
import argparse
import subprocess
import sys
import os

def run_command(cmd_list, log_file=None):
    print(f"Running: {' '.join(cmd_list)}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n\n--- Running: {' '.join(cmd_list)} ---\n")
            subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        # Stream to stdout if no log file
        subprocess.run(cmd_list, check=True)

def main(args):
    python_exe = sys.executable
    python_cmd = [python_exe, "-u"]
    
    # Ensure output directory exists (if output to same dir as ckpt)
    ckpt_dir = os.path.dirname(args.ckpt_path)
    
    # 1. Reconstruction (MSE / LPIPS)
    print("\n[1/6] Running Reconstruction Evaluation...")
    try:
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/eval_recon.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--data_root", args.data_root,
            "--max_items", str(args.num_samples),
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
        print(f"Error in Reconstruction: {e}")

    # 2. FID (Sample Quality)
    print("\n[2/6] Running FID Evaluation...")
    try:
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/eval_fid.py",
            "--ckpt_path", args.ckpt_path,
            "--prior", args.prior,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--data_root", args.data_root,
            "--num_samples", str(args.num_samples),
            "--steps", str(args.steps),
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in FID: {e}")

    # 3. Diversity
    print("\n[3/6] Running Diversity Evaluation...")
    try:
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/eval_diversity.py",
            "--ckpt_path", args.ckpt_path,
            "--prior", args.prior,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--num_samples", str(args.num_samples),
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in Diversity: {e}")

    # 4. Active Units
    print("\n[4/6] Running Active Units Evaluation...")
    try:
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/eval_active_units.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--data_root", args.data_root,
            "--num_samples", str(args.num_samples),
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in Active Units: {e}")

    # Define output suffix for Step-Dependent Priors
    suffix = ""
    if args.prior in ['imm', 'ddpm']:
        suffix = f"_steps{args.steps}"

    # 5. Interpolations
    print("\n[5/6] Generating Interpolations...")
    try:
        out_dir = os.path.join(ckpt_dir, f"viz_interp_manual{suffix}")
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/viz_interpolations.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--prior", args.prior,
            "--data_root", args.data_root,
            "--out_dir", out_dir,
            "--steps", str(args.steps),
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in Interpolations: {e}")

    # 6. Latent Structure (t-SNE)
    print("\n[6/6] Generating Latent Structure (t-SNE)...")
    try:
        out_dir = os.path.join(ckpt_dir, f"viz_structure_manual{suffix}")
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/viz_latent_structure.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--prior", args.prior,
            "--data_root", args.data_root,
            "--out_dir", out_dir,
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in t-SNE: {e}")
         
    # 7. Viz Reconstructions (Fixed Batch) - No Steps Dependency
    print("\n[7/8] Visualizing Reconstructions...")
    try:
        out_dir = os.path.join(ckpt_dir, "viz_recons")
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/viz_reconstructions.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--prior", args.prior,
            "--data_root", args.data_root,
            "--out_dir", out_dir,
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in Viz Reconstructions: {e}")

    # 8. Viz Samples (Fixed Seed)
    print("\n[8/8] Visualizing Samples...")
    try:
        out_dir = os.path.join(ckpt_dir, f"viz_samples{suffix}")
        run_command([
            *python_cmd, "e:/imm_vae/imm-main/viz_samples.py",
            "--ckpt_path", args.ckpt_path,
            "--dataset", args.dataset,
            "--model_type", args.model_type,
            "--prior", args.prior,
            "--out_dir", out_dir,
            "--steps", str(args.steps), # For IMM/DDPM
            "--z_dim", str(args.z_dim)
        ])
    except Exception as e:
         print(f"Error in Viz Samples: {e}")

    # --- Aggregate Results ---
    import json
    import pandas as pd
    from datetime import datetime

    print("\nAggregating Results...")
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": args.ckpt_path,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "prior": args.prior,
        "steps": args.steps,
        "z_dim": args.z_dim
    }

    # Load Recon Metrics
    try:
        with open(os.path.join(ckpt_dir, "metrics_recon.json"), "r") as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print("Warning: metrics_recon.json not found")

    # Load FID Metrics
    try:
        with open(os.path.join(ckpt_dir, "metrics_fid.json"), "r") as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print("Warning: metrics_fid.json not found")

    # Load Diversity Metrics
    try:
        with open(os.path.join(ckpt_dir, "metrics_diversity.json"), "r") as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print("Warning: metrics_diversity.json not found")

    # Load Active Units
    try:
        with open(os.path.join(ckpt_dir, "metrics_active_units.json"), "r") as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print("Warning: metrics_active_units.json not found")

    # Save to CSV
    csv_path = "e:/imm_vae/evaluation_results.csv"
    df = pd.DataFrame([results])
    
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
    print(f"\nAll evaluations completed. Results appended to {csv_path}")
    print("Numerical Results Summary:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation suite for a single model.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--prior', type=str, required=True, choices=['kl', 'mmd', 'imm', 'ddpm'], help='Prior type used in the model')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'CIFAR10', 'FashionMNIST', 'FFHQ'], help='Dataset name')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'nvae', 'vqvae'], help='Model architecture')
    parser.add_argument('--data_root', type=str, default='e:/imm_vae/data', help='Path to data root')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples for metrics')
    parser.add_argument('--steps', type=int, default=50, help='Number of sampling steps (for IMM/DDPM) if needed')
    
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    
    args = parser.parse_args()
    main(args)
