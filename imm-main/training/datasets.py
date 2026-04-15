
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        if not os.path.exists(root):
             print(f"Warning: Root {root} does not exist.")
             self.files = []
             return
             
        self.files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(dirpath, f))
        
        self.files.sort()
        # print(f"Found {len(self.files)} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
             img = Image.open(path).convert('RGB')
        except Exception as e:
             print(f"Error loading {path}: {e}")
             # Return random/dummy? Better to crash in training or skip.
             # Return the 0-th image recursively?
             return self.__getitem__(0)
             
        if self.transform:
            img = self.transform(img)
        # Return dummy label 0
        return img, 0
