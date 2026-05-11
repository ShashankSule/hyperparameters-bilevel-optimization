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


def biexponential_three_param(
    t: torch.Tensor,
    c1: torch.Tensor | float,
    lambda1: torch.Tensor | float,
    lambda2: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate c1 exp(-lambda1 t) + (1 - c1) exp(-lambda2 t)."""
    return c1 * torch.exp(-lambda1 * t) + (1.0 - c1) * torch.exp(-lambda2 * t)


def jacobian_biexp_three_param(
    t: torch.Tensor,
    c1: torch.Tensor | float,
    lambda1: torch.Tensor | float,
    lambda2: torch.Tensor | float,
) -> torch.Tensor:
    """Return the Jacobian with columns [dG/dc1, dG/dlambda1, dG/dlambda2]."""
    e1 = torch.exp(-lambda1 * t)
    e2 = torch.exp(-lambda2 * t)
    return torch.stack([e1 - e2, -c1 * t * e1, -(1.0 - c1) * t * e2], dim=1)


def biexponential_three_param_T2(
    t: torch.Tensor,
    c1: torch.Tensor | float,
    T21: torch.Tensor | float,
    T22: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate c1 exp(-t / T21) + (1 - c1) exp(-t / T22)."""
    return c1 * torch.exp(-t / T21) + (1.0 - c1) * torch.exp(-t / T22)


def jacobian_biexp_three_param_T2(
    t: torch.Tensor,
    c1: torch.Tensor | float,
    T21: torch.Tensor | float,
    T22: torch.Tensor | float,
) -> torch.Tensor:
    """Return the Jacobian with columns [dG/dc1, dG/dT21, dG/dT22]."""
    e1 = torch.exp(-t / T21)
    e2 = torch.exp(-t / T22)
    return torch.stack([e1 - e2, c1 * t * e1 / (T21**2), (1.0 - c1) * t * e2 / (T22**2)], dim=1)


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


def make_synthetic_observation_three_param(
    c1_true: float,
    lambda1_true: float,
    lambda2_true: float,
    t: torch.Tensor,
    sigma: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the noiseless and iid Gaussian-noisy three-parameter signals."""
    if seed is not None:
        set_noise_seed(seed)

    g_true = biexponential_three_param(t, c1_true, lambda1_true, lambda2_true)
    return g_true, add_gaussian_noise(g_true, sigma)


def make_synthetic_observation_three_param_T2(
    c1_true: float,
    T21_true: float,
    T22_true: float,
    t: torch.Tensor,
    sigma: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the noiseless and iid Gaussian-noisy three-parameter T2 signals."""
    if seed is not None:
        set_noise_seed(seed)

    g_true = biexponential_three_param_T2(t, c1_true, T21_true, T22_true)
    return g_true, add_gaussian_noise(g_true, sigma)
