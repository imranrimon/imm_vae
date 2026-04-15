import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# IMM/DDPM Research Suite (Two-Stage Pipeline)\n",
    "\n",
    "This notebook answers the following research questions:\n",
    "1. **Does IMM/DDPM improve sample quality (FID)?**\n",
    "2. **Does IMM/DDPM preserve reconstruction accuracy?**\n",
    "3. **Can IMM/DDPM learn meaningful latent geometry?**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from torchvision import transforms, datasets\n",
    "from torch.utils.data import DataLoader\n",
    "from tqdm import tqdm\n",
    "from scipy.linalg import sqrtm\n",
    "\n",
    "# Import Models\n",
    "from training.train_utils import set_seed, Model\n",
    "from training.imm_utils import IMM, ResidualDenoiser\n",
    "from training.ddpm_utils import DDPM\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Data & Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load CelebA Test Set\n",
    "image_size = 32\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((image_size, image_size)),\n",
    "    transforms.CenterCrop(image_size),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),\n",
    "])\n",
    "\n",
    "test_dataset = datasets.CelebA(root='./data/CelebA', split='test', download=False, transform=transform)\n",
    "test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)\n",
    "\n",
    "def load_models(z_dim=64, vae_path=\"checkpoints/best_vae.pth\", diffusion_path=\"checkpoints/best_diffusion.pth\"):\n",
    "    print(f\"Loading VAE from {vae_path}...\")\n",
    "    vae = Model(3, 32, 3, 128, z_dim, 32, 32).to(device)\n",
    "    if os.path.exists(vae_path):\n",
    "        vae.load_state_dict(torch.load(vae_path, map_location=device))\n",
    "    else:\n",
    "        print(\"Warning: VAE checkpoint not found!\")\n",
    "    vae.eval()\n",
    "\n",
    "    print(f\"Loading Diffusion from {diffusion_path}...\")\n",
    "    time_dim = 64\n",
    "    denoiser = ResidualDenoiser(\n",
    "        input_dim=z_dim + time_dim,\n",
    "        model_dim=1024,\n",
    "        output_dim=z_dim,\n",
    "        n_blocks=8\n",
    "    )\n",
    "    ddpm = DDPM(\n",
    "        network=denoiser,\n",
    "        z_dim=z_dim,\n",
    "        n_steps=1000,\n",
    "        device=device\n",
    "    ).to(device)\n",
    "    \n",
    "    if os.path.exists(diffusion_path):\n",
    "        ddpm.load_state_dict(torch.load(diffusion_path, map_location=device))\n",
    "    else:\n",
    "        print(\"Warning: Diffusion checkpoint not found!\")\n",
    "    ddpm.eval()\n",
    "    \n",
    "    return vae, ddpm\n",
    "\n",
    "vae_model, ddpm_model = load_models()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Reconstruction Accuracy (MSE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calc_reconstruction_mse(model, dataloader):\n",
    "    model.eval()\n",
    "    total_mse = 0.0\n",
    "    num_batches = 0\n",
    "    with torch.no_grad():\n",
    "        for x, _ in tqdm(dataloader, desc=\"Computing MSE\"):\n",
    "            x = x.to(device)\n",
    "\n",
    "            # VAE Forward\n",
    "            z, x_recon = model(x)\n",
    "            \n",
    "            # MSE\n",
    "            mse = torch.nn.functional.mse_loss(x_recon, x)\n",
    "            total_mse += mse.item()\n",
    "            num_batches += 1\n",
    "    return total_mse / num_batches\n",
    "\n",
    "# mse = calc_reconstruction_mse(vae_model, test_loader)\n",
    "# print(\"MSE:\", mse)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Sample Quality (FID)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torchvision.models import inception_v3\n",
    "from scipy.linalg import sqrtm\n",
    "\n",
    "def get_inception_features(loader, vae_model, ddpm_model=None, num_samples=2000, device=device):\n",
    "    inception = inception_v3(pretrained=True, transform_input=False).to(device)\n",
    "    inception.eval()\n",
    "    feats = []\n",
    "    count = 0\n",
    "    bs = 64\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        # If diffusion model present, generate samples\n",
    "        if ddpm_model is not None:\n",
    "            while count < num_samples:\n",
    "                curr_bs = min(bs, num_samples - count)\n",
    "                # 1. Sample z from DDPM\n",
    "                z_gen = ddpm_model.sample(curr_bs)\n",
    "                \n",
    "                # 2. Decode z with VAE\n",
    "                x_gen = vae_model.decoder(z_gen)\n",
    "                \n",
    "                # Resize for Inception\n",
    "                x_gen = torch.nn.functional.interpolate(x_gen, size=(299, 299), mode='bilinear')\n",
    "                f = inception(x_gen)\n",
    "                feats.append(f.cpu().numpy())\n",
    "                count += curr_bs\n",
    "\n",
    "        # Otherwise use real images\n",
    "        else:\n",
    "            for x, _ in loader:\n",
    "                x = x.to(device)\n",
    "                x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')\n",
    "                f = inception(x)\n",
    "                feats.append(f.cpu().numpy())\n",
    "                count += x.size(0)\n",
    "                if count >= num_samples: break\n",
    "    \n",
    "    if len(feats) > 0:\n",
    "        feats = np.concatenate(feats, axis=0)\n",
    "        return feats[:num_samples]\n",
    "    return np.array([])\n",
    "\n",
    "# Calculating FID (Pseudo-code as we need real features first)\n",
    "# real_feats = get_inception_features(test_loader, vae_model=None)\n",
    "# fake_feats = get_inception_features(None, vae_model, ddpm_model)\n",
    "# fid = calculate_fid(real_feats, fake_feats)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Latent Geometry (Interpolation & t-SNE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interpolation: Extract real z, interpolate, decode\n",
    "def visualize_interpolation(model, dataloader, device):\n",
    "    model.eval()\n",
    "    x, _ = next(iter(dataloader))\n",
    "    x = x.to(device)\n",
    "    z, _ = model(x)\n",
    "    z1 = z[0:1]\n",
    "    z2 = z[1:2]\n",
    "    alphas = torch.linspace(0, 1, 10).to(device)\n",
    "    images = []\n",
    "    for a in alphas:\n",
    "        z_interp = (1-a)*z1 + a*z2\n",
    "        recon = model.decoder(z_interp)\n",
    "        images.append(recon.cpu().squeeze())\n",
    "    fig, axs = plt.subplots(1, 10, figsize=(20, 2))\n",
    "    for i in range(10):\n",
    "        img = images[i].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5\n",
    "        axs[i].imshow(np.clip(img, 0, 1))\n",
    "        axs[i].axis('off')\n",
    "    plt.show()\n",
    "\n",
    "# t-SNE\n",
    "def visualize_tsne(model, dataloader, device, attribute_name='Male', num_samples=1000):\n",
    "    from sklearn.manifold import TSNE\n",
    "    attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', \n",
    "             'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', \n",
    "             'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', \n",
    "             'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', \n",
    "             'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', \n",
    "             'Wearing_Necktie', 'Young']\n",
    "    \n",
    "    if attribute_name not in attrs:\n",
    "        print(f\"Attribute {attribute_name} not found. Available: {attrs}\")\n",
    "        return\n",
    "        \n",
    "    attr_idx = attrs.index(attribute_name)\n",
    "    print(f\"Visualizing attribute: {attribute_name} (Index {attr_idx})\")\n",
    "    \n",
    "    model.eval()\n",
    "    latents = []\n",
    "    labels = []\n",
    "    count = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in tqdm(dataloader, desc=\"Extracting Latents\"):\n",
    "            x = x.to(device)\n",
    "            z, _ = model(x)\n",
    "            latents.append(z.cpu().numpy())\n",
    "            labels.append(y[:, attr_idx].cpu().numpy())\n",
    "            count += x.size(0)\n",
    "            if count >= num_samples: break\n",
    "    latents = np.concatenate(latents, axis=0)[:num_samples]\n",
    "    labels = np.concatenate(labels, axis=0)[:num_samples]\n",
    "    \n",
    "    tsne = TSNE(n_components=2, random_state=42)\n",
    "    z_2d = tsne.fit_transform(latents)\n",
    "    \n",
    "    plt.figure(figsize=(10, 8))\n",
    "    plt.scatter(z_2d[labels==0, 0], z_2d[labels==0, 1], c='blue', alpha=0.5, label=f'Not {attribute_name}', s=10)\n",
    "    plt.scatter(z_2d[labels==1, 0], z_2d[labels==1, 1], c='red', alpha=0.5, label=f'{attribute_name}', s=10)\n",
    "    plt.legend()\n",
    "    plt.title(f\"t-SNE of Latent Space - {attribute_name}\")\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("e:\\imm_vae\\Research_Suite.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print("Created Research_Suite.ipynb")
