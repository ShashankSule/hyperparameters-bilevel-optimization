"""Data generation helpers for biexponential recovery experiments."""

from __future__ import annotations

import numpy as np
import torch


def biexponential(t: torch.Tensor, c: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Evaluate c0 exp(-lambda0 t) + c1 exp(-lambda1 t)."""
    return c[0] * torch.exp(-lam[0] * t) + c[1] * torch.exp(-lam[1] * t)


def jacobian_biexp(t: torch.Tensor, c: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Return the Jacobian with columns [dG/dc0, dG/dc1, dG/dlambda0, dG/dlambda1]."""
    e0 = torch.exp(-lam[0] * t)
    e1 = torch.exp(-lam[1] * t)
    return torch.stack([e0, e1, -c[0] * t * e0, -c[1] * t * e1], dim=1)


def add_gaussian_noise(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add iid Gaussian noise with standard deviation sigma."""
    return signal + sigma * torch.randn_like(signal)


def add_rician_noise(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Rician noise by perturbing real and imaginary channels."""
    nr = sigma * torch.randn_like(signal)
    ni = sigma * torch.randn_like(signal)
    return torch.sqrt((signal + nr) ** 2 + ni**2)


def make_range_grid(
    range_config: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a uniformly spaced tensor from a {min, max, samples} config."""
    return torch.linspace(
        float(range_config["min"]),
        float(range_config["max"]),
        int(range_config["samples"]),
        dtype=dtype,
        device=device,
    )


def make_time_grid(
    config: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the measurement grid from config['t']."""
    return make_range_grid(config["t"], dtype=dtype, device=device)


def make_lambda_grid(
    range_config: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a lambda grid from a {min, max, samples} config."""
    return make_range_grid(range_config, dtype=dtype, device=device)


def set_noise_seed(seed: int) -> None:
    """Seed torch and numpy before generating a noise realization."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def make_synthetic_observation(
    c_true: torch.Tensor,
    lam_true: torch.Tensor,
    t: torch.Tensor,
    noise_type: str,
    sigma: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the noiseless signal and one noisy observation."""
    if seed is not None:
        set_noise_seed(seed)

    g_true = biexponential(t, c_true, lam_true)
    if noise_type == "gaussian":
        y = add_gaussian_noise(g_true, sigma)
    elif noise_type == "rician":
        y = add_rician_noise(g_true, sigma)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    return g_true, y
