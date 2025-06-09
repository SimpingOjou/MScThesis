import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=1):
        """
            LSTM Autoencoder for time series data.
            Expected input shape: (B, T, F), where B is batch size, T is sequence length, and F is feature dimension.
            Args:
                input_dim (int): Dimension of the input features (F).
                hidden_dim (int): Dimension of the hidden state in LSTM (H).
                latent_dim (int): Dimension of the latent space representation (L).
                num_layers (int): Number of LSTM layers.
        """
        super(LSTMAutoencoder, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LSTM device: {self._device}")

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, device=self._device)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim, device=self._device)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim, device=self._device)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, device=self._device)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        """
            Forward pass of the LSTM Autoencoder.
            Args:
                x (torch.Tensor): Input tensor of shape (B, T, F).
                lengths (torch.Tensor): Lengths of each sequence in the batch (B).
            Returns:
                out (torch.Tensor): Output tensor of shape (B, T, F).
                z (torch.Tensor): Latent representation of shape (B, L).
        """
        x = x.to(self._device)
        lengths = lengths.to(torch.device("cpu"))  # Ensure lengths are on CPU for packing
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # B, T, F
        encoded, _ = self.encoder(packed) # B, T, H
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True) # B, T, H

        # Apply linear to the last hidden output for each sequence
        last_hidden = unpacked[torch.arange(len(lengths)), lengths - 1] # B, H
        z = self.hidden_to_latent(last_hidden) # B, L

        # Decoder: project latent back and expand to full time
        h_dec = self.latent_to_hidden(z).unsqueeze(1).repeat(1, x.size(1), 1) # B, T, H
        out, _ = self.decoder(h_dec) # B, T, F

        return out, z # B, T, F and B, L
    
class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VAE device: {self._device}")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Latent variables
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM (will receive repeated z)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, lengths):
        x = x.to(self._device)
        lengths = lengths.to(torch.device("cpu"))  # Ensure lengths are on CPU for packing
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)  # h_n: (1, B, H)
        h_n = h_n.squeeze(0)

        mu = self.fc_mu(h_n)        # (B, L)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)

        # Expand z across time for decoding
        max_len = x.size(1)
        z_expanded = z.unsqueeze(1).repeat(1, max_len, 1)  # (B, T, L)

        dec_out, _ = self.decoder(z_expanded)
        x_hat = self.output_layer(dec_out)  # (B, T, F)
        return x_hat, mu, logvar # B, T, F - B, L - L
    
class LSTMVAE_t(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VAE_t device: {self._device}")
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu_t, logvar_t):
        std = torch.exp(0.5 * logvar_t)
        eps = torch.randn_like(std)
        return mu_t + eps * std

    def forward(self, x, lengths):
        x = x.to(self._device)
        lengths = lengths.to(torch.device("cpu"))  # Ensure lengths are on CPU for packing
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder_lstm(packed)
        h_t, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1)) # (B, T, H)

        # Project to latent space
        mu_t = self.fc_mu(h_t)        # (B, T, L)
        logvar_t = self.fc_logvar(h_t) # (B, T, L)
        z_t = self.reparameterize(mu_t, logvar_t)

        # Decode z_t
        packed_z = nn.utils.rnn.pack_padded_sequence(z_t, lengths, batch_first=True, enforce_sorted=False)
        packed_dec, _ = self.decoder_lstm(packed_z)
        h_dec, _ = nn.utils.rnn.pad_packed_sequence(packed_dec, batch_first=True, total_length=x.size(1))

        x_hat = self.output_layer(h_dec)  # (B, T, F)
        return x_hat, mu_t, logvar_t # shape (B, T, F), (B, T, L), (B, T, L)