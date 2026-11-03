"""MeanFlow latent priors (base + improved) for few-step / one-step sampling.

Convention (standard flow-matching, NOT the inverted one used in imm_core.py):

    z_t = (1 - t) * x + t * eps        t = 0 -> data,  t = 1 -> noise
    v(z_t, t) = dz_t/dt = eps - x      instantaneous velocity

MeanFlow models the *average* velocity over an interval instead of the
instantaneous one:

    u(z, r, t) = 1/(t - r) * integral_r^t v(z_tau, tau) d_tau

Because z_t - z_r = (t - r) * u(z_t, r, t), sampling is a single jump:

    z_r = z_t - (t - r) * u(z_t, r, t)

so r=0, t=1 gives a genuine 1-NFE sample.

Training uses the MeanFlow Identity, which needs no teacher and no distillation:

    u(z, r, t) = v(z, t) - (t - r) * d/dt u(z, r, t)

with the total derivative taken along the trajectory at fixed r:

    d/dt u = (d_z u) . v + d_t u

evaluated with one JVP (tangent (v, 0, 1)). The right-hand side is stop-gradded
and used as the regression target. When r == t the identity degenerates to
ordinary flow matching (u = v), which is why mixing in r == t samples anchors
training.

Two variants are provided:

  MeanFlow(variant='base')
      uniform (r, t), exact JVP, plain MSE.

  MeanFlow(variant='improved')
      logit-normal t sampling, a configurable fraction of r == t samples,
      adaptive per-sample loss weighting w = 1/(||delta||^2 + c)^p, and an
      optional JVP-free finite-difference estimate of the total derivative.

Provenance note: the base formulation above (identity, JVP target, 1-step
sampler) follows Geng et al., "Mean Flows for One-step Generative Modelling"
(2025). The 'improved' switches collect refinements reported for MeanFlow-style
training -- logit-normal time sampling, r == t mixing, adaptive weighting and
JVP-free total derivatives. Treat the exact hyper-parameters here as sensible
defaults for a 64-d latent, not as reproductions of any specific paper's
settings; check against the source before citing numbers.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device) * (-math.log(10000.0) / (half - 1))
        )
        args = x[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        return x + self.fc2(h)


class MeanFlowNet(nn.Module):
    """u_theta(z, r, t). Trunk matched to the IMM/DDPM denoiser (512 wide, 6 blocks)."""

    def __init__(self, z_dim=64, model_dim=512, n_blocks=6, time_dim=64,
                 cond_dim=0, time_scale=1.0):
        super().__init__()
        self.cond_dim = cond_dim
        # NOT 1000. MeanFlow's identity differentiates through this embedding
        # w.r.t. t, so a large input scale makes d(emb)/dt enormous and the
        # target v - (t-r)*du/dt is swamped by embedding-derivative noise.
        # Measured on an 8-mode ring: scale 1000 -> kurtosis +0.04 (Gaussian,
        # broken); scale 1-10 -> -1.06/-1.13 against a true -1.467.
        self.time_scale = time_scale
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.project_in = nn.Sequential(
            nn.Linear(z_dim + 2 * time_dim + cond_dim, model_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResBlock(model_dim) for _ in range(n_blocks)])
        self.project_out = nn.Sequential(
            nn.LayerNorm(model_dim), nn.SiLU(), nn.Linear(model_dim, z_dim)
        )

    def forward(self, z, r, t, cond=None):
        r = r.reshape(-1)
        t = t.reshape(-1)
        parts = [z, self.time_emb(r * self.time_scale), self.time_emb(t * self.time_scale)]
        if self.cond_dim:
            assert cond is not None, "conditional net called without cond"
            parts.append(cond)
        h = self.project_in(torch.cat(parts, dim=-1))
        for blk in self.blocks:
            h = blk(h)
        return self.project_out(h)


class MeanFlow(nn.Module):
    """MeanFlow prior over a flat latent vector.

    Always standardizes latents with the buffers below before modelling them,
    and de-standardizes on the way out -- the prior sees a roughly unit-scale
    target regardless of how the compressor's aggregate posterior is scaled.
    """

    def __init__(
        self,
        network: nn.Module,
        D: int = 64,
        variant: str = "base",
        data_mean=None,
        data_std=None,
        # improved-variant knobs
        p_ratio: float = 0.75,      # fraction of samples trained with r == t
        p_boundary: float = 0.2,    # fraction pinned to the sampler boundary (t=1 / r=0)
        logit_mu: float = -0.4,     # logit-normal location for t
        logit_sigma: float = 1.0,   # logit-normal scale for t
        adaptive_p: float = 0.5,    # exponent in w = 1/(loss + c)^p
        adaptive_c: float = 1e-3,
        jvp_free: bool = False,     # finite-difference instead of exact JVP
        fd_eps: float = 5e-3,
        target_clip: float = 0.0,   # clip ||target|| per sample; 0 disables
    ):
        super().__init__()
        assert variant in ("base", "improved"), variant
        self.network = network
        self.D = D
        self.variant = variant
        self.p_ratio = p_ratio
        self.p_boundary = p_boundary
        self.logit_mu = logit_mu
        self.logit_sigma = logit_sigma
        self.adaptive_p = adaptive_p
        self.adaptive_c = adaptive_c
        self.jvp_free = jvp_free
        self.fd_eps = fd_eps
        self.target_clip = target_clip
        self.conditional = getattr(network, 'cond_dim', 0) > 0

        self.register_buffer(
            "data_mean", data_mean if data_mean is not None else torch.zeros(D)
        )
        self.register_buffer(
            "data_std", data_std if data_std is not None else torch.ones(D)
        )

    # ------------------------------------------------------------ helpers
    def normalize(self, z):
        return (z - self.data_mean) / (self.data_std + 1e-6)

    def denormalize(self, z):
        return z * self.data_std + self.data_mean

    def forward(self, z, r, t, cond=None):
        return self.network(z, r, t, cond) if self.conditional else self.network(z, r, t)

    def _sample_times(self, B, device):
        """Returns (r, t) with r <= t, both in [0, 1], shaped [B, 1]."""
        if self.variant == "base":
            a = torch.rand(B, device=device)
            b = torch.rand(B, device=device)
            t = torch.maximum(a, b)
            r = torch.minimum(a, b)
        else:
            # logit-normal t concentrates capacity in the mid-noise band, but the
            # sampler always *starts* at t = 1, so force a slice of the batch onto
            # the boundary or that regime is never trained.
            t = torch.sigmoid(
                self.logit_mu + self.logit_sigma * torch.randn(B, device=device)
            )
            t = torch.where(torch.rand(B, device=device) < self.p_boundary,
                            torch.ones_like(t), t)
            r = t * torch.rand(B, device=device)
            # ... and always *ends* at r = 0, likewise.
            r = torch.where(torch.rand(B, device=device) < self.p_boundary,
                            torch.zeros_like(r), r)
            # a fraction of the batch trains the plain flow-matching case r == t
            keep = torch.rand(B, device=device) < self.p_ratio
            r = torch.where(keep, t, r)
        return r.view(B, 1), t.view(B, 1)

    # ------------------------------------------------------------ training
    def compute_loss(self, x, cond=None):
        x = self.normalize(x)
        B = x.shape[0]
        device = x.device

        eps = torch.randn_like(x)
        r, t = self._sample_times(B, device)

        z = (1.0 - t) * x + t * eps
        v = eps - x  # instantaneous velocity of this path

        net = ((lambda z_, r_, t_: self.network(z_, r_, t_, cond))
               if self.conditional else self.network)
        if self.jvp_free:
            # forward finite difference along (v, 0, 1); no forward-mode AD needed.
            # t + h must stay inside [0, 1], so step backwards near the top of the
            # range and divide by the *signed* step actually taken.
            h = torch.where(t + self.fd_eps <= 1.0,
                            torch.full_like(t, self.fd_eps),
                            torch.full_like(t, -self.fd_eps))
            with torch.no_grad():
                u_hi = net(z + h * v, r, t + h)
                u_lo = net(z, r, t)
                dudt = (u_hi - u_lo) / h
            u = net(z, r, t)
        else:
            def fn(z_, r_, t_):
                return net(z_, r_, t_)

            u, dudt = torch.func.jvp(
                fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
            )

        target = (v - (t - r) * dudt).detach()
        if self.target_clip:
            # The identity's target can blow up when dudt is large, which is the
            # usual cause of MeanFlow training divergence. Rescale (don't
            # truncate) any sample whose target exceeds the budget.
            nrm = target.norm(dim=-1, keepdim=True)
            scale = (self.target_clip / nrm.clamp(min=1e-12)).clamp(max=1.0)
            target = target * scale

        delta = u - target
        if self.variant == "base":
            return (delta ** 2).mean()

        per_sample = (delta ** 2).mean(dim=-1)
        w = 1.0 / (per_sample.detach() + self.adaptive_c) ** self.adaptive_p
        return (w * per_sample).mean()

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n_samples: int, steps: int = 1, cond=None) -> torch.Tensor:
        """Integrate noise (t=1) to data (t=0) in `steps` jumps. NFE == steps."""
        device = next(self.parameters()).device
        if self.conditional:
            assert cond is not None, "conditional prior needs cond"
            n_samples = cond.shape[0]
        z = torch.randn(n_samples, self.D, device=device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = torch.full((n_samples, 1), times[i], device=device)
            r = torch.full((n_samples, 1), times[i + 1], device=device)
            u = (self.network(z, r, t, cond) if self.conditional
                 else self.network(z, r, t))
            z = z - (t - r) * u
        return self.denormalize(z)

    # convenience so callers can share the IMM/DDPM sampling call shape
    @torch.no_grad()
    def sampling_times(self, steps: int, device=None):
        return torch.linspace(1.0, 0.0, steps + 1, device=device)
