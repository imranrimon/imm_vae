"""Retrain the MSE-NVAE Stage 1 on the CORRECTED CelebA crop, from cache.

Why: Section 2 of the paper states that the MSE-stack numbers were produced
before the CelebA transform was fixed (it resized to square and then
centre-cropped, making the crop a no-op) and that we never compare them
numerically to corrected-crop numbers. The diagnostic table breaks that promise:
it places the MSE-NVAE row beside the adversarial-AE row and the text reads "the
gap grows by a factor of five (42 -> 235)" straight across the two
preprocessings. Retraining on the corrected crop is what makes that comparison
legitimate.

Why not just run main.py --stage 1: that path re-decodes CelebA JPEGs every
epoch and runs the GPU at ~6% utilisation. This feeds the SAME training code
from the pre-decoded uint8 cache instead. The cache is built by the same
get_dataloader with the same transform, and the uint8 round-trip is exact
(ToTensor yields k/255; the cache stores k; to_float returns 2k/255 - 1, which
is what Normalize(0.5, 0.5) produces), so this is a speed change only.

The original checkpoint is untouched: this writes under the _newcrop run tag.

Usage:
    python train_mse_nvae_newcrop.py --epochs 100
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
import types

import torch

from train_stage1_fast import image_cache, to_float


class CachedLoader:
    """Yields (x, dummy_label) batches from an in-memory uint8 tensor."""

    def __init__(self, x_u8, batch_size, device, shuffle=True, drop_last=True):
        self.x = x_u8
        self.bs = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        stop = n - self.bs + 1 if self.drop_last else n
        for i in range(0, stop, self.bs):
            b = idx[i:i + self.bs]
            yield to_float(self.x[b]).to(self.device, non_blocking=True), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CelebA')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--run_tag', default='_newcrop')
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2])
    ap.add_argument('--cpu_cache', action='store_true',
                    help='keep the image cache in host memory; slower, '
                         'but avoids GPU pressure when sharing the device')
    ap.add_argument('--method', default=None,
                    help='Stage 2 prior: imm / ddpm / meanflow')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    what = ('Stage 1 MSE-NVAE' if a.stage == 1
            else f'Stage 2 {(a.method or "imm").upper()} prior')
    print(f"device {dev} | {what} on the CORRECTED {a.dataset} crop", flush=True)

    Xtr = image_cache(a.dataset, 'train', a.image_size)
    Xva = image_cache(a.dataset, 'test', a.image_size)
    print(f"cache: train {tuple(Xtr.shape)}  test {tuple(Xva.shape)}", flush=True)
    # Stage 2 additionally holds the frozen NVAE, the prior and its optimiser, and
    # re-encodes every batch; parking the whole 500 MB cache on the GPU on top of
    # that is what got the first attempt OOM-killed. Keep it in host memory and
    # ship batches across instead -- the transfer is not the bottleneck.
    if a.cpu_cache:
        print("  (cache stays on CPU, batches transferred per step -- slower)",
              flush=True)
    else:
        Xtr = Xtr.to(dev)
        Xva = Xva.to(dev)

    def fake_get_dataloader(dataset_name, batch_size=128, image_size=32,
                            split='train', root=None, drop_last=None):
        x = Xva if split == 'test' else Xtr
        # honour drop_last when the caller asks for it: IMM needs the batch
        # size to stay divisible by its particle-group size M.
        dl = True if drop_last is None else bool(drop_last)
        return CachedLoader(x, batch_size, dev, shuffle=(split != 'test'),
                            drop_last=dl)

    args = types.SimpleNamespace(
        stage=a.stage, model='nvae', dataset=a.dataset,
        method='kl' if a.stage == 1 else (a.method or 'imm'),
        epochs=a.epochs, z_dim=64, kl_weight=None, base_method='kl',
        lr=None, jvp_free=False, z_fine_ch=None, free_bits=0.0,
        kl_warmup=0, run_tag=a.run_tag)

    if a.stage == 1:
        import src.training.stage1 as S1
        S1.get_dataloader = fake_get_dataloader
        S1.train_stage1(args, dev)
        ck = f"checkpoints/nvae_{a.dataset}_kl{a.run_tag}_stage1.pth"
    else:
        import src.training.stage2 as S2
        S2.get_dataloader = fake_get_dataloader
        S2.train_stage2(args, dev)
        ck = (f"checkpoints/nvae_{a.dataset}_{args.method}"
              f"{a.run_tag}_stage2.pth")
    print(f"\ndone. checkpoint: {ck}", flush=True)


if __name__ == '__main__':
    main()
