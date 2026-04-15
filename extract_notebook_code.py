import json
import sys

def extract_code(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"--- Code from {notebook_path} ---")
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if source.strip():
                    print(f"\n# Cell {i+1}")
                    print(source)
                    print("-" * 20)
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_notebook_code.py <notebook_path>")
    else:
        for path in sys.argv[1:]:
            extract_code(path)
