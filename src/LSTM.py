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
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False) # B, T, F
        encoded, _ = self.encoder(packed) # B, T, H
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True) # B, T, H

        # Apply linear to the last hidden output for each sequence
        last_hidden = unpacked[torch.arange(len(lengths)), lengths - 1] # B, H
        z = self.hidden_to_latent(last_hidden) # B, L

        # Decoder: project latent back and expand to full time
        h_dec = self.latent_to_hidden(z).unsqueeze(1).repeat(1, x.size(1), 1) # B, T, H
        out, _ = self.decoder(h_dec) # B, T, F

        return out, z # B, T, F and B, L