
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, hidden_dims=[32, 64, 128, 256], start_sz=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        modules = []
        c_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(c_in, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            c_in = h_dim
        self.encoder_net = nn.Sequential(*modules)
        
        # Calculate flat features size
        # 32 -> 16 -> 8 -> 4 -> 2
        self.final_sz = start_sz // (2 ** len(hidden_dims))
        self.last_h_dim = hidden_dims[-1]
        self.flat_dim = self.last_h_dim * self.final_sz * self.final_sz
        
        print(f"VAE Init: hidden_dims={hidden_dims}, flat_dim={self.flat_dim}, final_sz={self.final_sz}")
        
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        
        modules = []
        # Create a copy to reverse, so self.hidden_dims (if stored) isn't affected? 
        # Actually hidden_dims is local arg.
        dec_hidden = hidden_dims[::-1]
        
        for i in range(len(dec_hidden) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_hidden[i], dec_hidden[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dec_hidden[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder_net = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dec_hidden[-1], dec_hidden[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dec_hidden[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(dec_hidden[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        result = self.encoder_net(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.last_h_dim, self.final_sz, self.final_sz)
        result = self.decoder_net(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
    def encoder(self, x):
        # Alias for consistency with training scripts
        return self.encode(x)
        
    def decoder(self, z):
        # Alias
        return self.decode(z)
