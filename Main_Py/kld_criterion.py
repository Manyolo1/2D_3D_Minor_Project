import torch
import torch.nn as nn


class KLDCriterion(nn.Module):
    """KL Divergence loss for VAE with a configurable coefficient.

    Computes: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per sample, then
    sums across batch and scales by kld_coeff.
    """

    def __init__(self, kld_coeff: float = 1.0):
        super().__init__()
        self.kld_coeff = float(kld_coeff) if kld_coeff is not None else 1.0

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL term per sample (sum across latent dims)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # Sum across batch (train_vae normalizes later when reporting)
        kld = torch.sum(kld_per_sample)
        return self.kld_coeff * kld
