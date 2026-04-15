import json

def parse_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(f"Cell {i}:\n{source}\n{'-'*20}\n")
    
    return "\n".join(code_cells)

if __name__ == "__main__":
    content = parse_notebook("e:/imm_vae/KL_vae_celebA.ipynb")
    with open("e:/imm_vae/extracted_code.txt", "w", encoding="utf-8") as f:
        f.write(content)

