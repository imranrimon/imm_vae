"""VQ autoencoder + an autoregressive prior over its code grid.

The third rung of the Stage-1 ladder. A discrete latent changes the character of
the prior--posterior gap entirely: there is no Gaussian to mismatch, so the
"prior mismatch" framing does not even apply in its usual form. The natural
free baseline is instead sampling each code independently from its marginal
frequency, and the natural learned prior is autoregressive over the grid.

Encoder and decoder are the ones from ldm_ae, so the only difference from the
continuous model is the quantizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ldm_ae import Encoder, Decoder


class VectorQuantizer(nn.Module):
    """Straight-through VQ with EMA-free (loss-based) codebook updates."""

    def __init__(self, n_codes=1024, dim=4, beta=0.25):
        super().__init__()
        self.n_codes, self.dim, self.beta = n_codes, dim, beta
        self.embed = nn.Embedding(n_codes, dim)
        self.embed.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z):
        # z: [B, C, H, W] -> flatten to [BHW, C]
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        d = (zf.pow(2).sum(1, keepdim=True)
             - 2 * zf @ self.embed.weight.t()
             + self.embed.weight.pow(2).sum(1))
        idx = d.argmin(1)
        zq = self.embed(idx).view(B, H, W, C).permute(0, 3, 1, 2)

        codebook_loss = F.mse_loss(zq, z.detach())
        commit_loss = F.mse_loss(z, zq.detach())
        loss = codebook_loss + self.beta * commit_loss

        zq = z + (zq - z).detach()          # straight-through
        return zq, loss, idx.view(B, H, W)

    def lookup(self, idx):
        B, H, W = idx.shape
        return self.embed(idx.reshape(-1)).view(B, H, W, self.dim).permute(0, 3, 1, 2)


class VqAutoencoder(nn.Module):
    def __init__(self, base=128, z_ch=4, n_res=2, n_codes=1024, quantizer='loss',
                 **qkw):
        super().__init__()
        self.z_ch, self.n_codes = z_ch, n_codes
        self.encoder = Encoder(base=base, z_ch=z_ch, n_res=n_res)
        self.decoder = Decoder(base=base, z_ch=z_ch, n_res=n_res)
        if quantizer == 'ema':
            from src.models.vq_ema import VectorQuantizerEMA
            self.quant = VectorQuantizerEMA(n_codes=n_codes, dim=z_ch, **qkw)
        else:
            self.quant = VectorQuantizer(n_codes=n_codes, dim=z_ch)

    def encode(self, x):
        # Encoder emits 2*z_ch (mu, logvar); VQ uses the mu half only.
        mu, _ = self.encoder(x)
        return mu

    def forward(self, x):
        z = self.encode(x)
        zq, vq_loss, idx = self.quant(z)
        return self.decoder(zq), vq_loss, idx

    def decode_codes(self, idx):
        return self.decoder(self.quant.lookup(idx))

    def last_layer(self):
        return self.decoder.conv_out.weight


class CodeTransformer(nn.Module):
    """Small autoregressive transformer over a flattened code grid.

    64 tokens for an 8x8 grid, so a compact decoder-only stack is enough; this
    is the discrete analogue of the conv prior used on continuous latents.
    """

    def __init__(self, n_codes=1024, seq=64, dim=256, heads=8, layers=6):
        super().__init__()
        self.n_codes, self.seq = n_codes, seq
        self.tok = nn.Embedding(n_codes + 1, dim)     # +1 = BOS
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_codes)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, idx):
        """idx: [B, seq] of code indices. Returns logits for next-token at each pos."""
        B, L = idx.shape
        bos = torch.full((B, 1), self.n_codes, device=idx.device, dtype=torch.long)
        inp = torch.cat([bos, idx[:, :-1]], dim=1)
        h = self.tok(inp) + self.pos[:, :L]
        h = self.body(h, mask=self.mask[:L, :L])
        return self.head(self.norm(h))

    def loss(self, idx):
        logits = self(idx)
        return F.cross_entropy(logits.reshape(-1, self.n_codes), idx.reshape(-1))

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0, top_k=0):
        idx = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(self.seq):
            k = idx.shape[1]
            pad = torch.zeros(n, self.seq - k, dtype=torch.long, device=device)
            # forward() prepends BOS and drops the last token, so output
            # position j carries p(token_j | tokens_<j). Having generated k
            # tokens we need position k, NOT k-1: the earlier off-by-one drew
            # every token from the preceding position's distribution, which is
            # why a model at NLL 2.03 still sampled like one at 5.34.
            logits = self(torch.cat([idx, pad], dim=1))[:, k]
            logits = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('inf')
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
