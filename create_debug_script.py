import re

def clean_and_fix_code(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove Cell headers and separators
    content = re.sub(r'Cell \d+:\n', '', content)
    content = re.sub(r'-{20,}\n', '', content)

    # Fix the logging line
    old_log = 'f"Reconstruction={nll.item():.5f}  MMD={kl.item():.5f}  " #IMM={imm_loss.item():.5f}'
    new_log = 'f"Reconstruction={nll.item():.5f}  KL={kl.item():.5f}  MMD={mmd.item():.5f}"'
    
    if old_log in content:
        content = content.replace(old_log, new_log)
    else:
        print("WARNING: Could not find the logging line to replace!")

    # Set steps to print more often for debug
    content = content.replace('print_every=100,', 'print_every=10,')
    content = content.replace('print_every = 200,', 'print_every = 10,')
    
    # Inject break for quick run within the train function loop
    # Finding the loop structure is tricky with regex, simpler to inject check in the loop
    # Searching for: "for images, _labels in dataloader:"
    loop_start = "for images, _labels in dataloader:"
    loop_inject = "for images, _labels in dataloader:\n            if step_idx > 20: break"
    content = content.replace(loop_start, loop_inject)

    # Reduce epochs in main
    content = content.replace('n_epochs = 80,', 'n_epochs = 1,')

    # Comment out show_reconstructions call to avoid GUI/Display issues in headless env if needed
    # But user is on windows, maybe ok. Safest to disable plotting for metrics verification.
    content = content.replace('show_reconstructions(model', '#show_reconstructions(model')
    content = content.replace('plt.show()', '#plt.show()')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    clean_and_fix_code("e:/imm_vae/extracted_code.txt", "e:/imm_vae/debug_training.py")
