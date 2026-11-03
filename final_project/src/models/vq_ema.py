"""Vector quantizer with the standard anti-collapse machinery.

Our first VQ autoencoder used loss-based codebook updates and ended with 194 of
1024 codes alive (81% dead) after 25 epochs, climbing ~7 codes/epoch. The
generative prior then could not recover what the quantizer had discarded. The
fixes below are the established ones; each is switchable so we can attribute the
recovery rather than ship a bundle and guess.

  EMA codebook       van den Oord et al. (2017), appendix A.1. Update the
                     codebook by an exponential moving average of the encoder
                     vectors assigned to it, instead of by gradient on a
                     codebook loss. Removes the dependence on the optimizer's
                     learning rate, which is what starves rarely-hit codes.

  k-means init       Seed the codebook from real encoder outputs in the first
                     batch rather than from a uniform prior. A codebook
                     initialized off-manifold has most entries permanently
                     farther from every encoder vector than the few that landed
                     nearby, which is the collapse mechanism at t=0.

  dead-code restart  Any code whose EMA cluster size falls below a threshold is
                     reinitialized to a randomly chosen encoder vector from the
                     current batch. Standard practice in large VQ models.

  L2 / cosine        ViT-VQGAN (Yu et al., 2021): normalize encoder outputs and
                     codebook entries to the unit sphere so lookup is by cosine
                     similarity. Removes the norm mismatch that lets a few
                     large-norm codes dominate assignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes=1024, dim=4, beta=0.25, decay=0.99, eps=1e-5,
                 l2_norm=True, restart=True, restart_thresh=1.0,
                 kmeans_init=True):
        super().__init__()
        self.n_codes, self.dim, self.beta = n_codes, dim, beta
        self.decay, self.eps = decay, eps
        self.l2_norm, self.restart = l2_norm, restart
        self.restart_thresh = restart_thresh

        embed = torch.randn(n_codes, dim)
        if l2_norm:
            embed = F.normalize(embed, dim=-1)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(n_codes))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('inited', torch.tensor(0 if kmeans_init else 1))

    # ---------------------------------------------------------------- helpers
    def _maybe_init(self, zf):
        if int(self.inited) == 1:
            return
        n = zf.shape[0]
        idx = torch.randperm(n, device=zf.device)[:self.n_codes]
        seed = zf[idx]
        if seed.shape[0] < self.n_codes:                    # pad by resampling
            extra = zf[torch.randint(0, n, (self.n_codes - seed.shape[0],),
                                     device=zf.device)]
            seed = torch.cat([seed, extra], 0)
        if self.l2_norm:
            seed = F.normalize(seed, dim=-1)
        self.embed.copy_(seed)
        self.embed_avg.copy_(seed)
        self.cluster_size.fill_(1.0)
        self.inited.fill_(1)

    def _codebook(self):
        return F.normalize(self.embed, dim=-1) if self.l2_norm else self.embed

    # ---------------------------------------------------------------- forward
    def forward(self, z):
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        zq_in = F.normalize(zf, dim=-1) if self.l2_norm else zf

        if self.training:
            self._maybe_init(zq_in)

        emb = self._codebook()
        d = (zq_in.pow(2).sum(1, keepdim=True)
             - 2 * zq_in @ emb.t()
             + emb.pow(2).sum(1))
        idx = d.argmin(1)
        onehot = F.one_hot(idx, self.n_codes).type(zf.dtype)
        zq = emb[idx].view(B, H, W, C).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                # EMA statistics are kept in fp32: under bf16 autocast the
                # encoder vectors are bf16, and an index-put into an fp32
                # buffer with a bf16 source raises.
                zin32 = zq_in.float()
                oh32 = onehot.float()
                n_i = oh32.sum(0)
                dw = oh32.t() @ zin32
                self.cluster_size.mul_(self.decay).add_(n_i, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = ((self.cluster_size + self.eps)
                      / (n + self.n_codes * self.eps) * n)
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))

                if self.restart:
                    dead = self.cluster_size < self.restart_thresh
                    k = int(dead.sum())
                    if k > 0:
                        pick = zin32[torch.randint(0, zin32.shape[0], (k,),
                                                   device=zin32.device)]
                        self.embed[dead] = pick
                        self.embed_avg[dead] = pick
                        self.cluster_size[dead] = 1.0

        # commitment only -- the codebook is no longer updated by gradient
        loss = self.beta * F.mse_loss(zf, zq.permute(0, 2, 3, 1).reshape(-1, C).detach())
        zq = z + (zq - z).detach()
        return zq, loss, idx.view(B, H, W)

    def lookup(self, idx):
        B, H, W = idx.shape
        emb = self._codebook()
        return emb[idx.reshape(-1)].view(B, H, W, self.dim).permute(0, 3, 1, 2)

    @torch.no_grad()
    def usage(self):
        """Fraction of codes with a non-trivial EMA cluster size."""
        return float((self.cluster_size > self.restart_thresh).float().mean())
