import pandas as pd
import numpy as np
import torch

def segment_steps_by_phase(df, phase_col="phase"):
    """
    Segments a DataFrame into steps based on the specified phase column.
    """
    phases = df[phase_col]

    # 1. Find where phase changes
    changes = phases != phases.shift()
    change_points = df.index[changes]

    # 2. Get the phases at these change points
    change_phases = phases.loc[change_points].reset_index(drop=True)

    # 3. Identify "stance" → ... → "swing" → "stance" sequences
    segments = []
    i = 0
    while i < len(change_phases) - 2:
        if change_phases[i] == "stance":
            swing_found = False
            for j in range(i+1, len(change_phases)):
                if change_phases[j] == "swing":
                    swing_found = True
                elif swing_found and change_phases[j] == "stance":
                    # # Skip first and last segment
                    # if i == 0 or j == len(change_phases) - 1:
                    #     i = j - 1
                    #     break
                    start_idx = change_points[i]
                    end_idx = change_points[j] - 1
                    segment = df.loc[start_idx:end_idx].drop(columns=["Step Phase Forelimb", "Step Phase Hindlimb"], errors='ignore')
                    
                    ## Reset x to 0
                    pose_cols = [col for col in segment.columns if 'pose' in col]
                    segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                    segments.append(segment)
                    i = j - 1  # skip ahead
                    break
            else:
                # no closing stance; go to end
                start_idx = change_points[i]
                segment = df.loc[start_idx:].drop(columns=["Step Phase Forelimb", "Step Phase Hindlimb"], errors='ignore')
                ## Reset x to 0
                pose_cols = [col for col in segment.columns if 'pose' in col]
                segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                segments.append(segment)
                break
        i += 1

    return segments

def steps_to_tensor(step_dicts, scaler):
    """
        Convert a list of step dictionaries to a padded tensor of shape (num_steps, max_length, num_features).

        Returns:
            - A tensor of shape (num_steps, max_length, num_features) containing the scaled step data (B, T, F).
            - A tensor of lengths for each step indicating the actual length of each step (B).
    """
    step_arrays = [scaler.transform(sd["step"].values) for sd in step_dicts]
    lengths = [len(step) for step in step_arrays]
    max_len = max(lengths)
    dim = step_arrays[0].shape[1]

    padded = np.zeros((len(step_arrays), max_len, dim), dtype=np.float32)
    for i, arr in enumerate(step_arrays):
        padded[i, :len(arr)] = arr

    return torch.tensor(padded), torch.tensor(lengths)

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

def masked_vae_t_loss(x_hat, x, lengths, mu_t, logvar_t, beta=1.0, free_bits=0.5):
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

    return recon_loss + beta * kl_loss

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