
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
             img = Image.open(path).convert('RGB')
        except Exception as e:
             print(f"Error loading {path}: {e}")
             return self.__getitem__(0)
             
        if self.transform:
            img = self.transform(img)
        return img, 0

def get_dataloader(dataset_name, batch_size=128, image_size=32, split='train', root='E:/imm_vae/data', drop_last=None):
    """
    Unified Data Loader Factory.
    Args:
        dataset_name (str): 'CelebA', 'CIFAR10', 'FFHQ'
        split (str): 'train' or 'test'
    """
    
    if dataset_name == 'CelebA':
        # CelebA standard transform
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Note: torchvision CelebA expects separate 'train', 'valid', 'test' splits
        target_split = 'train' if split == 'train' else 'test'
        # User path: E:\imm_vae\data\CelebA
        # datasets.CelebA expects 'root' to contain the files.
        dset = datasets.CelebA(root=os.path.join(root, 'CelebA'), split=target_split, download=False, transform=transform)
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        is_train = (split == 'train')
        dset = datasets.CIFAR10(root=os.path.join(root, 'CIFAR10'), train=is_train, download=False, transform=transform)
        
    elif dataset_name == 'FFHQ':
        # FFHQ usually simple folder structure
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # FFHQ doesn't have standard splits in simple folder, we might just use same folder
        dset = FlatImageFolder(root=os.path.join(root, 'FFHQ'), transform=transform)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    
    if drop_last is None:
        drop_last = (split == 'train')
        
    loader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=(split=='train'), 
        drop_last=drop_last, 
        num_workers=2, 
        pin_memory=True
    )
    return loader
