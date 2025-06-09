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