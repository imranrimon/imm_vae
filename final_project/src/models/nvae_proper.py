"""A faithful-in-spirit NVAE: deep hierarchy, bidirectional inference, residual
posteriors.

The project's original "NVAE" had two latent groups, a single conv conditional
prior, and an independently parameterized posterior. That is not what makes NVAE
work. The three components that do, from Vahdat & Kautz (2020):

  Bidirectional inference   One bottom-up encoder pass produces per-scale
                            features; a single top-down pass then generates the
                            groups in order. The SAME top-down stream feeds both
                            the prior p(z_l | z_<l) and the posterior
                            q(z_l | z_<l, x), so the two cannot drift apart
                            structurally.

  Residual normal posterior q predicts (dmu, dlogsigma) RELATIVE to the prior's
                            parameters, not absolute ones. At initialization
                            q == p and every KL term is zero, which is what
                            prevents both posterior collapse and the runaway
                            prior mismatch a from-scratch posterior suffers.

  Learned conditional prior Every group after the first is Gaussian conditioned
                            on the top-down state. Ancestral sampling from the
                            model is therefore ALREADY a learned hierarchical
                            prior -- there is no fixed N(0, I) to mismatch
                            except at the topmost group.

That last point is why this rung matters for our diagnostic: NVAE closes the
prior--posterior gap architecturally rather than by a second stage, so the
prediction is that a Stage-2 prior buys very little on top of it.

Also implements KL warm-up and per-group KL balancing, both of which NVAE needs
to keep deep hierarchies from collapsing their upper groups.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-excite, used throughout NVAE's residual cells."""

    def __init__(self, ch, r=8):
        super().__init__()
        h = max(ch // r, 4)
        self.fc1, self.fc2 = nn.Linear(ch, h), nn.Linear(h, ch)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(swish(self.fc1(s))))
        return x * s[:, :, None, None]


def _norm(kind, ch, groups=8):
    """BatchNorm keeps running statistics, which are estimated under
    posterior-driven activations during training and are therefore wrong when
    the model is sampled ancestrally -- worth 22-76 FID on our NVAE. GroupNorm
    has no running statistics, so the mismatch cannot arise."""
    if kind == 'group':
        return nn.GroupNorm(min(groups, ch), ch)
    return nn.BatchNorm2d(ch)


class Cell(nn.Module):
    """Residual cell: Norm-Swish-Conv x2 + SE, as in NVAE's encoder/decoder cells."""

    def __init__(self, ch, norm='batch'):
        super().__init__()
        self.bn1, self.bn2 = _norm(norm, ch), _norm(norm, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.se = SE(ch)

    def forward(self, x):
        h = self.c1(swish(self.bn1(x)))
        h = self.c2(swish(self.bn2(h)))
        return x + self.se(h)


class ProperNVAE(nn.Module):
    """Hierarchy of `groups_per_scale` latent groups at each of `scales`
    resolutions, coarsest first."""

    def __init__(self, in_ch=3, base=64, z_ch=8, scales=3, groups_per_scale=2,
                 img=32, norm='batch'):
        """groups_per_scale may be an int (same at every scale) or a per-scale
        list, which may contain zeros.

        Zero is necessary for a compressive hierarchy: a latent group at full
        resolution costs z_ch * img^2 dimensions, so at 32x32 any group on the
        finest scale is >= 1024 dims on its own. Placing groups only on coarse
        scales and upsampling deterministically afterwards is what lets the
        hierarchy match the 256-dim budget of the other compressors.
        """
        super().__init__()
        self.norm_kind = norm
        if isinstance(groups_per_scale, int):
            groups_per_scale = [groups_per_scale] * scales
        assert len(groups_per_scale) == scales, groups_per_scale
        self.gps_list = list(groups_per_scale)
        self.z_ch, self.scales = z_ch, scales
        self.gps = self.gps_list[0]
        self.img = img
        self.n_groups = sum(self.gps_list)

        # ---- bottom-up encoder: img -> img/2 -> img/4 (features per scale) ---
        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.down = nn.ModuleList()
        for i in range(scales - 1):
            self.down.append(nn.Sequential(
                Cell(base, norm), nn.Conv2d(base, base, 3, 2, 1)))
        self.enc_top = Cell(base, norm)

        # ---- top-down: one learned constant, then per-group prior/posterior --
        self.h0 = nn.Parameter(torch.zeros(1, base, img // 2 ** (scales - 1),
                                           img // 2 ** (scales - 1)))
        nn.init.normal_(self.h0, std=0.02)

        self.prior = nn.ModuleList()      # p(z_l | z_<l): 2*z_ch from top-down
        self.post = nn.ModuleList()       # q residual: 2*z_ch from [top-down, enc]
        self.merge = nn.ModuleList()      # fold z_l back into the top-down state
        self.cells = nn.ModuleList()
        for _ in range(self.n_groups):
            self.prior.append(nn.Conv2d(base, 2 * z_ch, 3, 1, 1))
            self.post.append(nn.Conv2d(2 * base, 2 * z_ch, 3, 1, 1))
            self.merge.append(nn.Conv2d(base + z_ch, base, 3, 1, 1))
            self.cells.append(Cell(base, norm))
        self.up = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          nn.Conv2d(base, base, 3, 1, 1))
            for _ in range(scales - 1)])
        self.out = nn.Sequential(_norm(norm, base),
                                 nn.Conv2d(base, in_ch, 3, 1, 1))

    # ---------------------------------------------------------------- helpers
    def _encode(self, x):
        """Bottom-up: returns features at each scale, coarsest last."""
        h = self.stem(x)
        feats = []
        for d in self.down:
            feats.append(h)
            h = d(h)
        feats.append(self.enc_top(h))
        return feats[::-1]                # coarsest first

    @staticmethod
    def _kl(dmu, dlv, lv_p):
        """KL between a residual normal and its prior.

        q = N(mu_p + dmu*sigma_p, (sigma_p*exp(dlv))^2) gives a KL that depends
        only on the residual terms -- zero when the encoder predicts nothing.
        """
        return 0.5 * (dmu.pow(2) + dlv.exp().pow(2) - 1.0 - 2.0 * dlv)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        feats = self._encode(x)
        B = x.shape[0]
        h = self.h0.expand(B, -1, -1, -1)
        kls, gi = [], 0
        for s in range(self.scales):
            ef = feats[s]
            for _ in range(self.gps_list[s]):
                mu_p, lv_p = self.prior[gi](h).chunk(2, 1)
                lv_p = lv_p.clamp(-8, 8)
                dmu, dlv = self.post[gi](torch.cat([h, ef], 1)).chunk(2, 1)
                dlv = dlv.clamp(-8, 8)
                mu_q = mu_p + dmu * torch.exp(lv_p)
                lv_q = lv_p + dlv
                z = mu_q + torch.randn_like(mu_q) * torch.exp(lv_q)
                kls.append(self._kl(dmu, dlv, lv_p).flatten(1).sum(1))
                h = self.cells[gi](self.merge[gi](torch.cat([h, z], 1)))
                gi += 1
            if s < self.scales - 1:
                h = self.up[s](h)
        return torch.tanh(self.out(h)), kls

    @torch.no_grad()
    def sample(self, n, device, t=1.0):
        """Ancestral sampling from the model's OWN hierarchical prior.

        `t` may be a scalar or a per-group sequence. Mixed sampling shows the
        coarse groups are free and the fine groups carry all the loss, so a
        single global temperature over- or under-damps somewhere.
        """
        ts = [t] * self.n_groups if not isinstance(t, (list, tuple)) else list(t)
        h = self.h0.expand(n, -1, -1, -1)
        gi = 0
        for s in range(self.scales):
            for _ in range(self.gps_list[s]):
                mu_p, lv_p = self.prior[gi](h).chunk(2, 1)
                lv_p = lv_p.clamp(-8, 8)
                z = mu_p + ts[gi] * torch.randn_like(mu_p) * torch.exp(lv_p)
                h = self.cells[gi](self.merge[gi](torch.cat([h, z], 1)))
                gi += 1
            if s < self.scales - 1:
                h = self.up[s](h)
        return torch.tanh(self.out(h))

    # ------------------------------------------------- Stage-2 hooks
    @torch.no_grad()
    def encode_top(self, x):
        """Posterior sample of the FIRST latent group.

        Every later group is conditioned on the top-down state, so group 1 is
        the only genuinely unconditional one and therefore the only place a
        Stage-2 prior can be inserted without displacing NVAE's own conditional
        priors.
        """
        feats = self._encode(x)
        h = self.h0.expand(x.shape[0], -1, -1, -1)
        mu_p, lv_p = self.prior[0](h).chunk(2, 1)
        lv_p = lv_p.clamp(-8, 8)
        dmu, dlv = self.post[0](torch.cat([h, feats[0]], 1)).chunk(2, 1)
        dlv = dlv.clamp(-8, 8)
        mu_q = mu_p + dmu * torch.exp(lv_p)
        lv_q = lv_p + dlv
        return mu_q + torch.randn_like(mu_q) * torch.exp(lv_q)

    @torch.no_grad()
    def sample_given_top(self, z1, t=1.0):
        """Ancestral sampling with group 1 supplied by an external prior."""
        n = z1.shape[0]
        h = self.h0.expand(n, -1, -1, -1)
        h = self.cells[0](self.merge[0](torch.cat([h, z1], 1)))
        gi = 1
        for s in range(self.scales):
            reps = self.gps_list[s] - 1 if s == 0 else self.gps_list[s]
            for _ in range(reps):
                mu_p, lv_p = self.prior[gi](h).chunk(2, 1)
                lv_p = lv_p.clamp(-8, 8)
                z = mu_p + t * torch.randn_like(mu_p) * torch.exp(lv_p)
                h = self.cells[gi](self.merge[gi](torch.cat([h, z], 1)))
                gi += 1
            if s < self.scales - 1:
                h = self.up[s](h)
        return torch.tanh(self.out(h))

    @torch.no_grad()
    def sample_mixed(self, x, k, t=1.0):
        """Groups 1..k from the PRIOR, groups k+1..L from the posterior of x.

        Sweeping k localises where ancestral sampling breaks: if the damage is
        concentrated at one transition, FID jumps there; if it accumulates, FID
        degrades smoothly.
        """
        feats = self._encode(x)
        B = x.shape[0]
        h = self.h0.expand(B, -1, -1, -1)
        gi = 0
        for si in range(self.scales):
            ef = feats[si]
            for _ in range(self.gps_list[si]):
                mu_p, lv_p = self.prior[gi](h).chunk(2, 1)
                lv_p = lv_p.clamp(-8, 8)
                if gi < k:                                  # prior
                    z = mu_p + t * torch.randn_like(mu_p) * torch.exp(lv_p)
                else:                                       # posterior
                    dmu, dlv = self.post[gi](torch.cat([h, ef], 1)).chunk(2, 1)
                    dlv = dlv.clamp(-8, 8)
                    mu_q = mu_p + dmu * torch.exp(lv_p)
                    lv_q = lv_p + dlv
                    z = mu_q + torch.randn_like(mu_q) * torch.exp(lv_q)
                h = self.cells[gi](self.merge[gi](torch.cat([h, z], 1)))
                gi += 1
            if si < self.scales - 1:
                h = self.up[si](h)
        return torch.tanh(self.out(h))

    @torch.no_grad()
    def recalibrate_bn(self, n=20000, bs=250, device='cuda', t=1.0):
        """Re-estimate BatchNorm running statistics under PRIOR sampling.

        BN stats are accumulated during training, where activations are driven
        by posterior samples. Ancestral sampling drives them differently, so the
        stored statistics are wrong exactly when generating. NVAE re-estimates
        them from prior samples before evaluation; without this the generative
        path is normalised by the wrong moments.
        """
        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        if not bns:                       # GroupNorm model: nothing to recalibrate
            return self
        for m in bns:
            m.reset_running_stats()
            m.momentum = None             # cumulative average
        self.train()
        for _ in range(n // bs):
            self.sample(bs, device, t=t)
        self.eval()
        return self

    def top_shape(self):
        r = self.img // 2 ** (self.scales - 1)
        return self.z_ch, r, r

    def latent_dims(self):
        d, r = 0, self.img // 2 ** (self.scales - 1)
        for s in range(self.scales):
            d += self.gps_list[s] * self.z_ch * r * r
            r *= 2
        return d
