import json

def fix_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    old_log = 'f"Reconstruction={nll.item():.5f}  MMD={kl.item():.5f}  " #IMM={imm_loss.item():.5f}\n'
    new_log = 'f"Reconstruction={nll.item():.5f}  KL={kl.item():.5f}  MMD={mmd.item():.5f}" #IMM={imm_loss.item():.5f}\n'
    
    fixed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if old_log.strip() in line.strip(): # Loose matching
                     new_source.append(line.replace(old_log.strip(), new_log.strip()))
                     fixed = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if not fixed:
        # Try stricter matching if loose failed, or exact string from extraction
        print("Loose match failed, trying exact replacement logic...")
        # Re-read to reset
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        target_str = 'f"Reconstruction={nll.item():.5f}  MMD={kl.item():.5f}  " #IMM={imm_loss.item():.5f}'
        replacement_str = 'f"Reconstruction={nll.item():.5f}  KL={kl.item():.5f}  MMD={mmd.item():.5f}"'
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    if target_str in line:
                        new_source.append(line.replace(target_str, replacement_str))
                        fixed = True
                    else:
                        new_source.append(line)
                cell['source'] = new_source

    if fixed:
        print("Fixed logging line in notebook.")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1) # Indent 1 to keep size reasonable but readable
    else:
        print("Could not find the line to fix!")

if __name__ == "__main__":
    fix_notebook("e:/imm_vae/KL_vae_celebA.ipynb")
