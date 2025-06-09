import torch
import os
import timeit
import pandas as pd
import numpy as np
import plotly.express as px

from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from src.Models import LSTMVAE_t

def masked_mse_loss(pred, target, lengths):
    """
        Compute the masked mean squared error loss between predicted and target tensors.
    """
    mask = torch.arange(target.size(1))[None, :].to(lengths.device) < lengths[:, None]
    mask = mask.unsqueeze(-1).expand_as(target)  # (B, T, F)
    mse = (pred - target) ** 2
    masked_mse = (mse * mask).sum() / mask.sum()
    return masked_mse

def masked_vae_loss(x_hat, x, lengths, mu, logvar, free_bits=0.1):
    """
    Masked VAE loss: reconstruction + KL divergence
    """
    B, T, F = x.shape
    mask = torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).expand_as(x)

    recon_loss = ((x_hat - x) ** 2) * mask
    recon_loss = recon_loss.sum() / mask.sum()

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    kl_div = torch.clamp(kl_div, min=0.1)  # where free_bits ~ 0.1

    return recon_loss + kl_div

def masked_vae_t_loss(x_hat, x, lengths, mu_t, logvar_t, free_bits=0.5):
    """
    Time-resolved VAE loss with masking support, per-unit free bits.
    """
    B, T, F = x.shape
    device = lengths.device

    # Compute mask
    mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).expand_as(x)  # (B, T, F)

    # Reconstruction loss
    recon_loss = ((x_hat - x) ** 2) * mask
    recon_loss = recon_loss.sum() / mask.sum()

    # KL divergence per latent unit
    kl_per_unit = -0.5 * (1 + logvar_t - mu_t.pow(2) - logvar_t.exp())  # (B, T, D)
    
    # Mask time steps (assume all D dimensions are valid per timepoint)
    time_mask = mask[..., 0]  # (B, T)

    # Clamp each latent unit to free bits
    kl_clamped = torch.clamp(kl_per_unit, min=free_bits)
    kl_loss = (kl_clamped * time_mask.unsqueeze(-1)).sum() / time_mask.sum()

    # Debug prints
    # kl_raw = (kl_per_unit * time_mask.unsqueeze(-1)).sum() / time_mask.sum()
    # tqdm.write(f"Reconstruction loss: {recon_loss.item():.4f}, Raw KL: {kl_raw.item():.4f}, Clamped KL: {kl_loss.item():.4f}, β: {beta:.3f}")

    return recon_loss + kl_loss

def load_model(model_path, input_dim, hidden_dim, latent_dim):
    """
    Loads a model from the given path.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMVAE_t(input_dim, hidden_dim, latent_dim)
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def train_model(
        step_tensor, lengths, input_dim, hidden_dim, latent_dim,
        batch_size, lr, num_epochs, patience, min_delta,
        models_dir
    ):
    """
    Trains the LSTMVAE_t model and saves the best model and loss plots.
    """

    train_idx, val_idx = train_test_split(np.arange(len(step_tensor)), test_size=0.2, random_state=42)
    train_data = TensorDataset(step_tensor[train_idx], lengths[train_idx])
    val_data = TensorDataset(step_tensor[val_idx], lengths[val_idx])

    def collate_fn(batch):
        x, lengths = zip(*batch)
        return torch.stack(x), torch.stack(lengths)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)

    model = LSTMVAE_t(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(models_dir, f'lstm_autoencoder_{now}.pt')

    val_losses = []
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        t1 = timeit.default_timer()
        model.train()
        train_loss = 0
        for batch_x, batch_lens in train_loader:
            batch_x, batch_lens = batch_x.to(device), batch_lens.to(device)
            x_hat, mu_t, logvar_t = model(batch_x, batch_lens)
            loss = masked_vae_t_loss(x_hat, batch_x, batch_lens, mu_t, logvar_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_lens in val_loader:
                batch_x, batch_lens = batch_x.to(device), batch_lens.to(device)
                x_hat, mu_t, logvar_t = model(batch_x, batch_lens)
                loss = masked_vae_t_loss(x_hat, batch_x, batch_lens, mu_t, logvar_t)
                val_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        train_losses.append(avg_train)
        t2 = timeit.default_timer()

        if avg_val < best_val_loss:
            print(f"Epoch {epoch+1}, Train: {avg_train:.4f} - Val: {avg_val:.4f} - Time: {t2-t1:.2f}s")
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = model.state_dict()

        early_stopper.step(avg_val)
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    torch.save({'model_state_dict': best_state,
                'epoch': best_epoch,
                'val_loss': best_val_loss},
                best_model_path)
    print(f"Best model saved at: {best_model_path} with val loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses

def save_model_losses(train_losses, val_losses, figures_dir, title_font_size, axis_title_font_size, legend_font_size):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    losses_df = pd.DataFrame({
        'epoch': np.arange(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    losses_df.to_csv(os.path.join(figures_dir, f'losses_{now}.csv'), index=False)
    fig = px.line(losses_df, x='epoch', y=['train_loss', 'val_loss'],
                    labels={'value': 'Loss', 'epoch': 'Epoch'},
                    title='Training and Validation Losses')
    fig.update_layout(title_font_size=title_font_size,
                        xaxis_title_font_size=axis_title_font_size,
                        yaxis_title_font_size=axis_title_font_size,
                        legend_font_size=legend_font_size)
    fig.show()

def load_and_plot_losses(losses_file, title_font_size, axis_title_font_size, legend_font_size):
    """
    Load losses from a CSV file and plot them.
    """
    losses_df = pd.read_csv(losses_file)
    fig = px.line(losses_df, x='epoch', y=['train_loss', 'val_loss'],
                    labels={'value': 'Loss', 'epoch': 'Epoch'},
                    title='Training and Validation Losses')
    fig.update_layout(title_font_size=title_font_size,
                        xaxis_title_font_size=axis_title_font_size,
                        yaxis_title_font_size=axis_title_font_size,
                        legend_font_size=legend_font_size)
    fig.show()
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True