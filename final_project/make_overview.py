"""Figure 1: the paper's framing, drawn from measured quantities only.

The thesis is that a learned latent prior pays off in proportion to the
prior-posterior gap, and that the gap is set by the compressor. That is three
claims, so the figure has three panels:

  (a) where Stage 2 acts, and what "the gap" names -- the only schematic panel.
  (b) the gap is a property of the compressor: the aggregate posterior's
      per-dimension standard deviation, measured, against the prior's 1.0.
  (c) the gap predicts the payoff: measured free-baseline FID against the
      measured gain of the learned prior over the best closed-form control.

Nothing here is drawn free-hand except the panel-(a) boxes; panels (b) and (c)
are computed from cached latents and results/*.csv at figure-build time.

Usage:
    python make_overview.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullFormatter, NullLocator

OUT = r'G:\imm_vae\paper\figures\overview.pdf'
INK, MUTE = '#1a1a1a', '#8a8a8a'
C_LOW, C_MID, C_HIGH = '#4878a8', '#c88a3a', '#b04a4a'


def per_dim_std(path, key='z'):
    d = torch.load(path, map_location='cpu')
    z = d[key] if isinstance(d, dict) and key in d else d
    if isinstance(z, dict):
        z = list(z.values())[0]
    return z.reshape(len(z), -1).float().std(0).numpy()


def kl_per_dim(path):
    """Mean over coordinates of KL(N(mu_i, s_i^2) || N(0, 1)), in nats.

    The paper's gap statistic: unlike the mean per-dimension std it registers a
    shifted mean and does not let over- and under-dispersed coordinates cancel.
    """
    d = torch.load(path, map_location='cpu')
    z = d['z'] if 'z' in d else d['train']
    z = z.reshape(len(z), -1).float()
    s, m = z.std(0), z.mean(0)
    return float((0.5 * (s ** 2 + m ** 2 - 1 - torch.log(s ** 2))).mean())


def gains():
    """control / learned for the three compressors, read from results CSVs."""
    import pandas as pd
    R = 'results/CelebA/'
    d = pd.read_csv(R + 'newcrop_diagnostic.csv')
    d = d[d.Variant == 'newcrop']
    mse = (d[d.Prior.str.contains('temperature')].FID_std.min()
           / float(d[d.Prior.str.contains('IMM')].FID_std.iloc[0]))
    c = pd.read_csv(R + 'ldm_controls.csv')
    sig_ctrl = c[c.Variant == 'closed-form-sigma'].FID_std.min()
    adv_ctrl = c[c.Variant.isna() | (c.Variant == 'closed-form')].FID_std.min()
    lp = pd.read_csv(R + 'ldm_prior.csv')
    sig_lrn = lp[(lp.Variant == 'w192b6_sigma')
                 & lp.Prior.str.contains('imm')].FID_std.min()
    sv = pd.read_csv('results/seed_variance.csv')
    adv_lrn = float(sv[(sv.Dataset == 'CelebA') & (sv.Res == 32)]['mean'].iloc[0])
    return mse, sig_ctrl / sig_lrn, adv_ctrl / adv_lrn


# --------------------------------------------------------------- panel (a) --
def panel_a(ax):
    ax.set_xlim(0, 10.1); ax.set_ylim(-0.45, 3.15); ax.axis('off')

    def box(x, w, label, sub, fc='white', ec=INK):
        ax.add_patch(FancyBboxPatch((x, 1.95), w, 0.68,
                                    boxstyle='round,pad=0.05', lw=1.1,
                                    facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, 2.40, label, ha='center', va='center',
                fontsize=7.6, color=INK)
        ax.text(x + w / 2, 2.14, sub, ha='center', va='center',
                fontsize=6.3, color=MUTE, style='italic')

    def arrow(x0, x1, y=2.29):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                                     mutation_scale=8, lw=1.0, color=INK))

    box(0.10, 1.45, r'$x$', 'image')
    arrow(1.60, 1.95)
    box(2.00, 2.45, 'Encoder', 'Stage 1')
    arrow(4.50, 5.05)
    box(5.10, 1.25, r'$z$', 'latent')
    arrow(6.40, 6.95)
    box(7.00, 2.45, 'Decoder', 'frozen')

    # the gap, hung underneath the latent
    ax.add_patch(FancyBboxPatch((2.30, 0.30), 6.10, 1.20,
                                boxstyle='round,pad=0.06', lw=1.0,
                                facecolor='#f7f4ef', edgecolor=MUTE, ls='--'))
    xs = np.linspace(-3.4, 3.4, 240)
    px = np.exp(-xs ** 2 / 2)
    qx = (0.62 * np.exp(-(xs - 1.15) ** 2 / (2 * 0.72 ** 2))
          + 0.38 * np.exp(-(xs + 1.35) ** 2 / (2 * 0.5 ** 2)))
    bx, by, bw, bh = 3.30, 0.46, 4.05, 0.72
    X = bx + (xs - xs.min()) / (xs.max() - xs.min()) * bw
    ax.plot(X, by + px / max(px) * bh, color=MUTE, lw=1.2)
    ax.plot(X, by + qx / max(qx) * bh, color=C_HIGH, lw=1.4)
    ax.fill_between(X, by + px / max(px) * bh, by + qx / max(qx) * bh,
                    color=C_HIGH, alpha=0.13)
    ax.text(bx + bw + 0.14, by + 0.60, r'$q(z)$', fontsize=7.2, color=C_HIGH)
    ax.text(bx + bw + 0.14, by + 0.22, r'$p(z)$', fontsize=7.2, color=MUTE)
    ax.text(2.52, 1.24, 'the gap', fontsize=7.0, color=INK, style='italic')
    ax.plot([5.72, 5.72], [1.56, 1.92], color=MUTE, lw=0.8, ls=':')
    ax.text(5.35, -0.16, 'Stage 2 learns a prior to close this',
            fontsize=7.0, color=INK, ha='center')
    ax.set_title('(a)  Where the gap lives', fontsize=8.5, loc='left',
                 color=INK, pad=5)


# --------------------------------------------------------------- panel (b) --
def panel_b(ax):
    # corrected-crop latents: the old cache predates the CelebA crop fix,
    # and pairing it with the post-fix rows below would mix preprocessings.
    rows = [('MSE NVAE', 'checkpoints/_latents_nvae_CelebA_newcrop.pt', C_LOW),
            (r'$\sigma$-VAE', 'checkpoints/_ldmae_latents_CelebA_4_sigma.pt', C_MID),
            ('Adversarial AE', 'checkpoints/_ldmae_latents_CelebA_4.pt', C_HIGH)]
    ax.axvline(1.0, color=MUTE, lw=1.0, ls='--', zorder=1)
    ax.text(1.03, 2.66, r'$p(z)$', fontsize=6.8, color=MUTE, va='top')
    for i, (lbl, path, col) in enumerate(rows):
        sd = per_dim_std(path)
        y = 2 - i
        ax.scatter(sd, y + np.random.RandomState(0).uniform(-.13, .13, len(sd)),
                   s=5, color=col, alpha=.45, lw=0, zorder=3)
        ax.plot([sd.min(), sd.max()], [y, y], color=col, lw=1.1, alpha=.85,
                zorder=2)
        ax.scatter([sd.mean()], [y], s=34, color=col, marker='|', lw=1.8,
                   zorder=4)
        ax.text(sd.max() + .10, y, f'[{sd.min():.2f}, {sd.max():.2f}]',
                fontsize=6.5, color=MUTE, ha='left', va='center')
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.set_ylim(-.7, 2.7); ax.set_xlim(0.42, 3.95)
    ax.set_xlabel(r'per-dimension std of $q(z)$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_title('(b)  The compressor sets the gap', fontsize=8.5, loc='left',
                 color=INK, pad=5)


# --------------------------------------------------------------- panel (c) --
def panel_c(ax, pts):
    for lbl, x, gain, col, mk, face in pts:
        ax.scatter([x], [gain], s=52, color=col, marker=mk, zorder=3,
                   facecolor=face if face else col, lw=1.3)
    off = {'MSE NVAE': (1.18, 0.88, 'left'),
           r'$\sigma$-VAE': (1.15, 1.10, 'left'),
           'Adversarial AE': (0.90, 0.72, 'right')}
    for lbl, x, gain, col, mk, face in pts:
        fx, fy, ha = off[lbl]
        ax.annotate(lbl, (x, gain), xytext=(x * fx, gain * fy),
                    fontsize=7.0, color=INK, ha=ha)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'gap: KL of $q(z)$ from prior (nats/dim)', fontsize=8)
    ax.set_ylabel(r'payoff: control / learned', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.015, 1.3); ax.set_ylim(0.92, 8.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.03, 0.1, 0.3, 1.0])
    ax.set_xticklabels(['0.03', '0.1', '0.3', '1'])
    ax.set_yticks([1, 2, 5])
    ax.set_yticklabels([r'$1\times$', r'$2\times$', r'$5\times$'])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=.18, lw=.6)
    ax.set_title('(c)  The gap predicts the payoff', fontsize=8.5, loc='left',
                 color=INK, pad=5)


def main():
    g_mse, g_sig, g_adv = gains()
    lat = {'mse': 'checkpoints/_latents_nvae_CelebA_newcrop.pt',
           'sig': 'checkpoints/_ldmae_latents_CelebA_4_sigma.pt',
           'adv': 'checkpoints/_ldmae_latents_CelebA_4.pt'}
    # MSE NVAE drawn open: not capacity-matched to the other two (8256 vs 256).
    pts = [('MSE NVAE', kl_per_dim(lat['mse']), g_mse, C_LOW, 'o', 'white'),
           (r'$\sigma$-VAE', kl_per_dim(lat['sig']), g_sig, C_MID, 'o', None),
           ('Adversarial AE', kl_per_dim(lat['adv']), g_adv, C_HIGH, 'o', None)]
    for lbl, x, g, *_ in pts:
        print(f'  panel (c): {lbl:16s} KL {x:.3f} nats/dim   gain {g:.3f}x')

    fig = plt.figure(figsize=(7.3, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.08, 0.80],
                          wspace=.55, left=.012, right=.985, top=.85, bottom=.20)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]), pts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight')
    fig.savefig(OUT.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print('wrote', OUT)
    for lbl, path, _ in [('MSE NVAE', 'checkpoints/_latents_nvae_CelebA_newcrop.pt', 0),
                         ('sigma-VAE', 'checkpoints/_ldmae_latents_CelebA_4_sigma.pt', 0),
                         ('adversarial AE', 'checkpoints/_ldmae_latents_CelebA_4.pt', 0)]:
        sd = per_dim_std(path)
        print(f'  {lbl:16s} mean {sd.mean():.3f}  [{sd.min():.3f}, {sd.max():.3f}]')


if __name__ == '__main__':
    main()
