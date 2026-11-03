"""LDM-style KL-regularized autoencoder (Stage 1 replacement).

The project's compressors are trained with pure summed MSE, which is why every
sample is blurry regardless of what the prior does -- the measured ceiling
(decoding TRUE encoder latents) is FID 5.9, and that ceiling is set entirely by
the decoder. This is the standard fix from Rombach et al.: an autoencoder with

    L1 reconstruction + LPIPS perceptual + PatchGAN adversarial + tiny KL

The KL weight is deliberately negligible (1e-6, as in LDM). The point of this
stage is a sharp, faithful decoder with a *diffusable* latent -- not a tight
variational bound. At 32x32 with f=4 the latent is 8x8xC.

Adversarial weighting follows LDM's adaptive rule: balance the GAN gradient
against the reconstruction gradient at the decoder's last layer, so the
discriminator never overwhelms fidelity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(c):
    return nn.GroupNorm(num_groups=min(32, c // 4 if c >= 4 else 1),
                        num_channels=c, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.norm1 = group_norm(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.norm2 = group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Encoder(nn.Module):
    """32x32 -> 8x8 (f=4), returns (mu, logvar) with z_ch channels."""

    def __init__(self, in_ch=3, base=128, z_ch=4, n_res=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base, 3, 1, 1)
        blocks = []
        ch = base
        for i, mult in enumerate([1, 2]):
            out = base * mult
            for _ in range(n_res):
                blocks.append(ResBlock(ch, out)); ch = out
            blocks.append(nn.Conv2d(ch, ch, 3, 2, 1))   # downsample
        self.blocks = nn.Sequential(*blocks)
        self.mid = nn.Sequential(ResBlock(ch, ch), ResBlock(ch, ch))
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, 2 * z_ch, 3, 1, 1)

    def forward(self, x):
        h = self.mid(self.blocks(self.conv_in(x)))
        h = self.conv_out(F.silu(self.norm_out(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, torch.clamp(logvar, -30.0, 20.0)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, base=128, z_ch=4, n_res=2):
        super().__init__()
        ch = base * 2
        self.conv_in = nn.Conv2d(z_ch, ch, 3, 1, 1)
        self.mid = nn.Sequential(ResBlock(ch, ch), ResBlock(ch, ch))
        blocks = []
        for mult in [2, 1]:
            out = base * mult
            for _ in range(n_res):
                blocks.append(ResBlock(ch, out)); ch = out
            blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            blocks.append(nn.Conv2d(ch, ch, 3, 1, 1))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def forward(self, z):
        h = self.blocks(self.mid(self.conv_in(z)))
        return torch.tanh(self.conv_out(F.silu(self.norm_out(h))))


class LdmAutoencoder(nn.Module):
    def __init__(self, base=128, z_ch=4, n_res=2):
        super().__init__()
        self.z_ch = z_ch
        self.encoder = Encoder(base=base, z_ch=z_ch, n_res=n_res)
        self.decoder = Decoder(base=base, z_ch=z_ch, n_res=n_res)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar, z

    def last_layer(self):
        return self.decoder.conv_out.weight


class PatchDiscriminator(nn.Module):
    """PatchGAN with GroupNorm + spectral norm. Shallow: 32x32 needs little depth.

    NOT BatchNorm. Real and fake are fed in separate forward passes, so
    BatchNorm normalizes each batch by its own statistics and erases exactly the
    difference the discriminator exists to detect -- observed here as the hinge
    loss pinned at exactly 1.000 (D outputting ~0 for both) while the generator
    kept getting a meaningless gradient. GroupNorm has no batch statistics, and
    spectral norm keeps the critic Lipschitz-bounded so it cannot run away.
    """

    def __init__(self, in_ch=3, base=64, n_layers=2):
        sn = nn.utils.spectral_norm
        super().__init__()
        layers = [sn(nn.Conv2d(in_ch, base, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(1, n_layers + 1):
            out = min(base * 2 ** i, 256)
            stride = 2 if i < n_layers else 1
            layers += [sn(nn.Conv2d(ch, out, 4, stride, 1)),
                       nn.GroupNorm(min(32, out // 4), out),
                       nn.LeakyReLU(0.2, True)]
            ch = out
        layers += [sn(nn.Conv2d(ch, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real, fake):
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


def adaptive_weight(nll_loss, g_loss, last_layer, max_w=1.0):
    """LDM's rule: scale the GAN term so its gradient matches the recon gradient.

    max_w is 1.0, not LDM's 1e4. With 1e4, a discriminator that stops
    discriminating (hinge loss pinned at 1.0, D output ~0) has vanishing
    gradients, so the ratio explodes to the clamp and a meaningless GAN term
    swamps reconstruction -- observed here as LPIPS regressing 0.044 -> 0.076
    the moment the GAN switched on.
    """
    nll_g = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_g = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    w = nll_g.norm() / (g_g.norm() + 1e-4)
    return torch.clamp(w, 0.0, max_w).detach()
