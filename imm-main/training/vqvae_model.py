
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.simple_ae import Encoder, Decoder

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, H, W]
        # Convert inputs to [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach() # Gradient copy
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert back to [B, C, H, W]
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAE, self).__init__()
        
        # We reuse the Encoder/Decoder from simple_ae but they are designed for flat mu/logvar output.
        # VQ-VAE usually needs a spatial map output.
        # simple_ae.Encoder output: (B, z_dim), (B, z_dim) - this is global average pooled.
        # We need SPATIAL output for VQ-VAE to work well.
        # So we cannot reuse simple_ae.Encoder directly as is.
        # We will define a simplified spatial encoder here or modify simple_ae further.
        # Modifying simple_ae.Encoder to return spatial map? 
        # simple_ae.Encoder has 'stats = self.mid(s3)'. 's3' is Res/4 feature map.
        # We can implement a new lightweight SpatialEncoder.
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_hiddens // 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1),
            nn.Conv2d(num_hiddens, embedding_dim, 1, 1) # Project to embedding dim
        )
        # Note: This is an 8x8 map if input is 32x32 (down 4x). 16x16 if 64x64.
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity

class VQVAE_Wrapper(nn.Module):
    """
    Wrapper to match the API expected by train_joint.py and eval scripts
    """
    def __init__(self, z_dim=128, input_res=64, in_channels=3):
        super().__init__()
        # Adapted parameters
        num_hiddens = 128
        num_embeddings = 512
        embedding_dim = 64 # Dimension of each vector in codebook
        commitment_cost = 0.25
        
        # Internal components
        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1),
            nn.Conv2d(num_hiddens, embedding_dim, 1, 1) 
        )
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder_net = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, in_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def encoder(self, x):
        """Compatible Encoder API: returns mean, logvar (dummy)"""
        z = self.encoder_net(x)
        # Return z and dummy logvar (zeros)
        return z, torch.zeros_like(z)

    def decoder(self, z):
        """Compatible Decoder API: Quantizes then Decodes"""
        # We assume z is continuous (from encoder or prior)
        # Quantize (ignore loss/perplexity here, just get quantized latents)
        # VectorQuantizer returns: loss, quantized, perplexity, encodings
        # We can extract just quantized.
        # But _vq_vae expects inputs [B, C, H, W].
        _, quantized, _, _ = self._vq_vae(z)
        x_rec = self.decoder_net(quantized)
        return x_rec

    def forward(self, x):
        z = self.encoder_net(x)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self.decoder_net(quantized)
        
        return quantized, x_recon, vq_loss, perplexity

