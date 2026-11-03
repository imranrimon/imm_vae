"""IMM latent prior.

KNOWN CONVENTION DEFECT -- read before trusting Stage 2 IMM numbers.

compute_loss interpolates as

    x_t = t * x + (1 - t) * z_prior          (t = 1 -> DATA, t = 0 -> NOISE)

i.e. the time axis is inverted relative to standard OT-FM / IMM, where t = 1 is
noise. But it also draws s ~ U(tmin, t), so s < t, which was copied from the
standard convention where s < t means "s is closer to the data". Under the
inverted axis used here, s < t means s is closer to the NOISE, so
f(x_t, s, t) is trained as a *noising* map, not a generative one.

Consequences:
  * Sampling on a descending grid (t: 1 -> tmin) runs the map in the direction
    it was trained, which walks toward noise, not data.
  * The original evaluation used an ASCENDING grid, which is out of the training
    domain but flips the sign of c_out = -(t - s) and therefore partially
    inverts the map. That is why the ascending ("legacy") sampler empirically
    scores better than the in-domain descending one on these checkpoints.

Two further inconsistencies follow from the same root cause:
  * OTFMSchedule.alpha(t) returns 1, but the interpolation above puts a
    coefficient of t on the data, so alpha_eff(t) = t. Everything derived from
    alpha -- c_in, lamb, lamb_diff, _weight_fn, _ddim -- is computed for a
    forward process that is not the one being used.
  * lamb(t) = -2 log(1 - t) therefore reports maximal SNR at t -> 1 and
    logSNR = 0 at t = 0, where the true SNR is zero.

Fixing this properly means flipping the interpolation to x_t = (1 - t) x + t eps
(and re-deriving alpha/sigma/eta accordingly), then RETRAINING. It cannot be
repaired at sampling time. See src/models/meanflow.py for a prior written with
the standard convention from the start.
"""
import torch
import torch.nn as nn
import math

# ────────────────────────────────────────────────────────────────
# Schedulers
# ────────────────────────────────────────────────────────────────
class IMMSchedule:
    def __init__(self):
        pass
    def alpha(self, t):
        raise NotImplementedError
    def sigma(self, t):
        raise NotImplementedError
    def lamb(self, t):
        return 2.0 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))
    def lamb_diff(self, t):
        raise NotImplementedError
    def eta(self, t):
        raise NotImplementedError
    def eta_inv(self, val):
        raise NotImplementedError

class OTFMSchedule(IMMSchedule):
    def __init__(self):
        super().__init__()
        self.tmin = 1e-5
        self.tmax = 1.0 - 1e-5

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return 1.0 - t

    def lamb(self, t):
        return -2.0 * torch.log(self.sigma(t))

    def lamb_diff(self, t):
        return 2.0 / (1.0 - t)

    def eta(self, t):
        return -torch.log(1.0 - t)

    def eta_inv(self, val):
        return 1.0 - torch.exp(-val)

# ────────────────────────────────────────────────────────────────
# Parameterization
# ────────────────────────────────────────────────────────────────
class IMMParameterization(nn.Module):
    def __init__(self, sigma_data: float, schedule: IMMSchedule):
        super().__init__()
        self.sigma_data = sigma_data
        self.schedule = schedule

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:
        return 1000.0 * t

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        a = self.schedule.alpha(t)
        s = self.schedule.sigma(t)
        return 1.0 / (self.sigma_data * torch.sqrt(a*a + s*s))

    def c_skip(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def c_out(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class OtFmEulerFmParameterization(IMMParameterization):
    def __init__(self, sigma_data: float):
        super().__init__(sigma_data, OTFMSchedule())

    def c_skip(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(s)

    def c_out(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -(t - s) * self.sigma_data

class OTFMScheduleFixed(IMMSchedule):
    """Standard OT-FM: x_t = (1-t) x + t eps.  t = 0 -> DATA, t = 1 -> NOISE.

    This is the corrected counterpart of OTFMSchedule, whose time axis is
    inverted (see the module docstring). Here alpha and sigma genuinely describe
    the interpolation used by compute_loss, so everything derived from them
    (c_in, lamb, lamb_diff, the loss weighting) is self-consistent, and s < t
    correctly means "s is closer to the data".
    """

    def __init__(self):
        super().__init__()
        self.tmin = 1e-3
        self.tmax = 1.0 - 1e-3

    def alpha(self, t):
        return 1.0 - t

    def sigma(self, t):
        return t

    def lamb(self, t):
        # logSNR = 2 log(alpha/sigma); decreasing in t (t=1 is noise)
        return 2.0 * (torch.log(1.0 - t) - torch.log(t))

    def lamb_diff(self, t):
        # d lamb / dt = -2 / (t (1-t))  -- negative, SNR falls as t rises
        return -2.0 / (t * (1.0 - t))

    def eta(self, t):
        # monotone increasing in noise level; eta = -lamb/2
        return torch.log(t) - torch.log(1.0 - t)

    def eta_inv(self, val):
        return torch.sigmoid(val)


class OtFmEulerFmParameterizationFixed(IMMParameterization):
    def __init__(self, sigma_data: float):
        super().__init__(sigma_data, OTFMScheduleFixed())

    def c_skip(self, s, t):
        return torch.ones_like(s)

    def c_out(self, s, t):
        # x_s = x_t - (t - s) * v, with t > s (t noisier). Matches flow matching.
        return -(t - s) * self.sigma_data


# ────────────────────────────────────────────────────────────────
# Residual Denoiser 
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# Residual Denoiser 
# ────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * 2)
        self.fc2  = nn.Linear(dim * 2, dim)
        self.act  = nn.SiLU()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h

class ResidualDenoiser(nn.Module):
    def __init__(self,
                 input_dim:  int,
                 model_dim:  int = 1024,
                 n_blocks:   int = 8,
                 output_dim: int = None):
        super().__init__()
        output_dim = output_dim or (input_dim - 2)
        self.project_in  = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResBlock(model_dim) for _ in range(n_blocks)])
        self.project_out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        h = self.project_in(x)
        for block in self.blocks:
            h = block(h)
        return self.project_out(h)


# ────────────────────────────────────────────────────────────────
# IMM Class
# ────────────────────────────────────────────────────────────────
class IMM(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 parameterization: IMMParameterization,
                 a: int = 1,
                 b: int = 4,
                 eta_max: float = 160.0,
                 eta_min: float = 1e-8,
                 k: int = 12,
                 M: int = 16,
                 D: int = 2,
                 data_mean = None,
                 data_std = None,
                 convention: str = 'legacy'):
        super().__init__()
        assert convention in ('legacy', 'fixed'), convention
        self.convention = convention
        self.network = network
        self.parameterization = parameterization
        self.schedule = parameterization.schedule

        self.D = D
        self.a = a
        self.b = b
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.k = k
        self.M = M
        
        # Register normalization buffers
        self.register_buffer('data_mean', data_mean if data_mean is not None else torch.zeros(D))
        self.register_buffer('data_std', data_std if data_std is not None else torch.ones(D))

    def forward(self, x_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        skip = self.parameterization.c_skip(s, t) * x_t  # [B, D]
        c_in_t = self.parameterization.c_in(t)       # [B, 1]
        x_scaled = x_t * c_in_t                       # [B, D]
        noise_s = self.parameterization.c_noise(s)    # [B, 1]
        noise_t = self.parameterization.c_noise(t)    # [B, 1]
        inp = torch.cat([x_scaled, noise_s, noise_t], dim=-1)  # [B, D+2]
        out = self.network(inp)                               # [B, D]
        out = self.parameterization.c_out(s, t) * out         # [B, D]
        return skip + out

    def _ddim(self, x_source, x_target, s, t):
        a_s = self.schedule.alpha(s)
        a_t = self.schedule.alpha(t)
        sig_s = self.schedule.sigma(s)
        sig_t = self.schedule.sigma(t)
        term1 = (a_s - (sig_s / sig_t) * a_t) * x_target
        term2 = (sig_s / sig_t) * x_source
        return term1 + term2

    def _kernel(self, x, y, s, t):
        eps = 1e-4
        c_out_st = self.parameterization.c_out(s, t)
        w = torch.abs(1.0 / (c_out_st + eps))
        
        diff = x - y
        dist = torch.sqrt(torch.sum(diff**2, dim=-1) + eps)
        
        if torch.isnan(dist).any(): print("NAN in dist")
        if torch.isnan(w).any(): print("NAN in w")

        w_flat = w.view_as(dist)
        log_k = -w_flat * dist / math.sqrt(self.D)
        
        mask = (t.view(-1) > s.view(-1))
        res = torch.where(mask, torch.exp(log_k), torch.zeros_like(log_k))
        if torch.isnan(res).any(): print("NAN in kernel result")
        return res

    def _kernel_for_diff(self, difference, s, t):
        zeros = torch.zeros_like(difference)
        return self._kernel(difference, zeros, s.unsqueeze(-1), t.unsqueeze(-1))

    def _weight_fn(self, s, t):
        t_flat = t.view(-1)
        a_t = self.schedule.alpha(t).view(-1)
        sig_t = self.schedule.sigma(t).view(-1)
        logSNR_t = self.schedule.lamb(t).view(-1)
        # dlogSNR_t is positive (2/(1-t)). 
        # Previous code had * (-dlogSNR_t), making weights negative.
        # We need positive weights for minimization.
        dlogSNR_t = self.schedule.lamb_diff(t).view(-1)
        # magnitude only: the corrected schedule has dlogSNR/dt < 0, and a
        # negative weight would turn the objective into a maximisation.
        coef = 0.5 * torch.sigmoid(self.b - logSNR_t) * dlogSNR_t.abs()
        denom = a_t.pow(2) + sig_t.pow(2)
        num = a_t.pow(self.a)
        return coef * num / denom

    def _map_fn(self, s, t):
        eps = (self.eta_max - self.eta_min) / (2 ** self.k)
        eta_t = self.schedule.eta(t)
        val = eta_t - eps
        inv = self.schedule.eta_inv(val)
        return torch.maximum(s, inv)

    def compute_loss(self, x):
        B = x.shape[0]
        M = self.M
        assert B % M == 0, f"Batch size ({B}) not divisible by M ({M})"
        G = B // M
        
        # Normalize Data to Standard Normal Dist
        x = (x - self.data_mean) / (self.data_std + 1e-6)
        
        z_prior = self.parameterization.sigma_data * torch.randn_like(x)
        t_range = self.schedule.tmax - self.schedule.tmin
        t_single = torch.rand((G, 1), device=x.device) * t_range + self.schedule.tmin
        s_single = torch.rand((G, 1), device=x.device) * (t_single - self.schedule.tmin) + self.schedule.tmin
        
        t = t_single.repeat_interleave(M, dim=0)
        s = s_single.repeat_interleave(M, dim=0)
        r = self._map_fn(s, t)
        
        # Interpolation. 'legacy' keeps the inverted axis (t=1 -> data) that the
        # existing checkpoints were trained under; 'fixed' uses standard OT-FM
        # (t=1 -> noise), which makes s < t mean "s is closer to the data" and
        # turns f(x_t, s, t) into a generative map instead of a noising one.
        if self.convention == 'fixed':
            x_t = (1.0 - t) * x + t * z_prior
            x_r = (1.0 - r) * x + r * z_prior
        else:
            x_t = t * x + (1.0 - t) * z_prior
            x_r = r * x + (1.0 - r) * z_prior
        
        f_x_t = self.forward(x_t, s, t)
        f_x_r = self.forward(x_r, s, r).detach()
        
        f_x_t = f_x_t.view(G, M, self.D)
        f_x_r = f_x_r.view(G, M, self.D)
        
        diff1 = (f_x_t.unsqueeze(1) - f_x_t.unsqueeze(2)).reshape(G*M*M, self.D)
        diff2 = (f_x_r.unsqueeze(1) - f_x_r.unsqueeze(2)).reshape(G*M*M, self.D)
        diff3 = (f_x_t.unsqueeze(1) - f_x_r.unsqueeze(2)).reshape(G*M*M, self.D)
        
        ss = s_single.repeat_interleave(M*M, dim=0).view(-1)
        ts = t_single.repeat_interleave(M*M, dim=0).view(-1)
        rs = self._map_fn(ss.unsqueeze(-1), ts.unsqueeze(-1)).view(-1)
        
        term1 = self._kernel_for_diff(diff1, ss, ts)
        term2 = self._kernel_for_diff(diff2, ss, rs)
        term3 = -2.0 * self._kernel_for_diff(diff3, ss, ts)
        
        weights = self._weight_fn(ss.unsqueeze(-1), ts.unsqueeze(-1))
        
        joint = term1 + term2 + term3
        loss = torch.mean(joint * weights)
        return loss
        
    def sampling_times(self, steps: int, device=None) -> torch.Tensor:
        """Descending time grid matching the training domain (s < t).

        compute_loss only ever draws s ~ U(tmin, t), so the network is trained
        exclusively for s < t. Sampling must therefore walk t downward.
        `steps` transitions require `steps + 1` grid points.

        NOTE: this makes the sampler consistent with the training domain, but
        because the time axis itself is inverted (see the module docstring) that
        direction walks toward noise. On checkpoints trained with the current
        compute_loss this scores WORSE than the original ascending grid. The
        real fix is to correct the convention and retrain.
        """
        if self.convention == 'fixed':
            # t = 1 is noise, t = 0 is data: walk t downward to generate.
            return torch.linspace(self.schedule.tmax, self.schedule.tmin,
                                  steps + 1, device=device)
        return torch.linspace(1.0, self.schedule.tmin, steps + 1, device=device)

    def sample(self, n_samples: int, times: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        times = times.to(device)
        if times[0] < times[-1]:
            raise ValueError(
                "IMM.sample expects a descending time grid (t > s at every step); "
                "got an ascending one. Use IMM.sampling_times(steps)."
            )
        x = self.parameterization.sigma_data * torch.randn((n_samples, self.D), device=device)
        for i in range(len(times) - 1):
            t_val = times[i]
            s_val = times[i + 1]
            t_tensor = torch.full((n_samples, 1), t_val, device=device)
            s_tensor = torch.full((n_samples, 1), s_val, device=device)
            x = self.forward(x, s_tensor, t_tensor)

        # Denormalize Data back to Original Space
        x = x * self.data_std + self.data_mean
        return x

# ────────────────────────────────────────────────────────────────
# Simple DDPM Wrapper
# ────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleDDPM(nn.Module):
    def __init__(self, network: nn.Module, z_dim: int, time_dim: int = 128):
        super().__init__()
        self.network = network
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, t, s=None, class_labels=None):
        # x: [B, z_dim]
        # t: [B] or [B, 1] (usually random [0,1])
        
        if t.dim() == 2:
            t = t.squeeze(-1) # [B]
            
        # Scale t to [0, 1000] approx for embedding frequency
        t_scaled = t * 1000.0
        
        t_emb = self.time_mlp(t_scaled) # [B, time_dim]
        
        inp = torch.cat([x, t_emb], dim=-1) # [B, z_dim + time_dim]
        out = self.network(inp)
        return out
