"""Optimization routines for bilevel biexponential recovery experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import biexponential, jacobian_biexp

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for batch runs.
    tqdm = None


def sp(u: torch.Tensor) -> torch.Tensor:
    """Positive reparameterization used for amplitudes and decay rates."""
    return F.softplus(u)


def sp_d(u: torch.Tensor) -> torch.Tensor:
    """Derivative of softplus."""
    return torch.sigmoid(u)


def inv_sp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus for positive x."""
    x = x.clamp(min=1e-6)
    return x + torch.log(-torch.expm1(-x))


def default_x_init(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Default initialization [c0, c1, lambda0, lambda1]."""
    return torch.tensor([0.6, 0.4, 0.5, 2.0], dtype=dtype, device=device)


def _as_x_init(x_init: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
    if x_init is None:
        return default_x_init(dtype=t.dtype, device=t.device)
    return x_init.to(dtype=t.dtype, device=t.device)


def _eye4(reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(4, dtype=reference.dtype, device=reference.device)


def _iter_range(n_steps: int, desc: str, progress: bool):
    if progress and tqdm is not None:
        return tqdm(range(n_steps), desc=desc)
    return range(n_steps)


def lower_level_gn_ep(
    y: torch.Tensor,
    mu: float,
    t: torch.Tensor,
    beta: float = 0.0,
    x_star: torch.Tensor | None = None,
    max_iter: int = 300,
    tol: float = 1e-9,
    x_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """
    Solve the lower-level regularized nonlinear least-squares problem.

    The optimized variable is unconstrained, then mapped through softplus so
    amplitudes and decay rates stay positive.
    """
    x_init = _as_x_init(x_init, t)
    u = inv_sp(x_init.clamp(min=1e-3)).clone().detach()

    mu_eff = float(mu) + float(beta)
    if beta > 0.0 and x_star is not None:
        x_prior = (float(beta) / mu_eff) * x_star.to(dtype=t.dtype, device=t.device)
    else:
        x_prior = torch.zeros_like(x_init)

    eye4 = _eye4(t)
    hist = []
    for _ in range(max_iter):
        x = sp(u)
        c, lam = x[:2], x[2:]
        g_x = biexponential(t, c, lam)
        residual = y - g_x

        loss = (residual**2).sum() + mu_eff * ((x - x_prior) ** 2).sum()
        hist.append(loss.item())

        j_x = jacobian_biexp(t, c, lam)
        sd = sp_d(u)
        j_u = j_x * sd.unsqueeze(0)

        h_gn = j_u.T @ j_u + mu_eff * eye4 + 1e-7 * eye4
        grad = -j_u.T @ residual + mu_eff * (x - x_prior) * sd
        du = torch.linalg.solve(h_gn, -grad)

        step = 1.0
        for _ in range(25):
            u_new = u + step * du
            x_new = sp(u_new)
            residual_new = y - biexponential(t, x_new[:2], x_new[2:])
            loss_new = (residual_new**2).sum() + mu_eff * ((x_new - x_prior) ** 2).sum()

            if loss_new <= loss - 1e-4 * step * (grad * du).sum():
                break
            step *= 0.5

        u = (u + step * du).detach()
        if step * du.norm() < tol:
            break

    return sp(u).detach(), hist


class _XhatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_scalar, y, t, x_init):
        mu_val = mu_scalar.item()
        x_sol, _ = lower_level_gn_ep(
            y,
            mu_val,
            t,
            beta=0.0,
            x_star=torch.zeros_like(x_init),
            x_init=x_init,
        )
        c, lam = x_sol[:2], x_sol[2:]
        j_x = jacobian_biexp(t, c, lam)
        h = j_x.T @ j_x + mu_val * _eye4(t) + 1e-7 * _eye4(t)
        ctx.save_for_backward(x_sol, h)
        return x_sol

    @staticmethod
    def backward(ctx, grad_output):
        x_sol, h = ctx.saved_tensors
        z = torch.linalg.solve(h, x_sol)
        dmu = -(grad_output * z).sum().reshape(1)
        return dmu, None, None, None


class XhatModule(nn.Module):
    """Learnable log_mu wrapper returning xhat(mu) with implicit-diff backward."""

    def __init__(self, mu_init: float = 1.0):
        super().__init__()
        self.log_mu = nn.Parameter(torch.tensor(float(np.log(mu_init))))

    @property
    def mu(self) -> torch.Tensor:
        return torch.exp(self.log_mu)

    def forward(self, y: torch.Tensor, t: torch.Tensor, x_init: torch.Tensor | None = None) -> torch.Tensor:
        x_init = _as_x_init(x_init, t)
        return _XhatFn.apply(self.mu, y, t, x_init)


def gradient_descent_mu(
    y: torch.Tensor,
    t: torch.Tensor,
    mu_init: float = 1.0,
    lr: float = 0.15,
    n_steps: int = 100,
    lower_max_iter: int = 300,
    lower_tol: float = 1e-9,
    progress: bool = False,
) -> dict[str, list]:
    """Optimize log_mu with a manually computed implicit-differentiation gradient."""
    log_mu = torch.tensor(float(np.log(mu_init)), dtype=t.dtype, device=t.device)
    x_init = default_x_init(dtype=t.dtype, device=t.device)
    hist = {"mu": [], "V": [], "xhat": []}

    for _ in _iter_range(n_steps, "outer loop step", progress):
        mu_val = torch.exp(log_mu).item()
        xhat, _ = lower_level_gn_ep(
            y,
            mu_val,
            t,
            beta=0.0,
            x_star=torch.zeros_like(x_init),
            max_iter=lower_max_iter,
            tol=lower_tol,
            x_init=x_init,
        )
        c, lam = xhat[:2], xhat[2:]
        g_x = biexponential(t, c, lam)
        residual = y - g_x
        j_x = jacobian_biexp(t, c, lam)
        h = j_x.T @ j_x + mu_val * _eye4(t) + 1e-7 * _eye4(t)

        z = torch.linalg.solve(h, xhat)
        j = j_x @ z
        d_v_dmu = 2 * (residual * j).sum()
        d_v_dlogmu = d_v_dmu * torch.exp(log_mu)
        log_mu = (log_mu - lr * d_v_dlogmu).detach()

        hist["mu"].append(torch.exp(log_mu).item())
        hist["V"].append((residual**2).sum().item())
        hist["xhat"].append(xhat)
        x_init = xhat

    return hist


def gauss_newton_mu(
    y: torch.Tensor,
    t: torch.Tensor,
    mu_init: float = 1.0,
    n_steps: int = 100,
    lower_max_iter: int = 300,
    lower_tol: float = 1e-9,
    progress: bool = False,
) -> dict[str, list]:
    """Optimize mu with a scalar Gauss-Newton step and positivity line search."""
    mu = torch.tensor(float(mu_init), dtype=t.dtype, device=t.device)
    x_init = default_x_init(dtype=t.dtype, device=t.device)
    hist = {"mu": [], "V": [], "xhat": []}

    for _ in _iter_range(n_steps, "outer loop step", progress):
        mu_val = mu.item()
        xhat, _ = lower_level_gn_ep(
            y,
            mu_val,
            t,
            beta=0.0,
            x_star=torch.zeros_like(x_init),
            max_iter=lower_max_iter,
            tol=lower_tol,
            x_init=x_init,
        )
        c, lam = xhat[:2], xhat[2:]
        g_x = biexponential(t, c, lam)
        residual = y - g_x
        j_x = jacobian_biexp(t, c, lam)
        h = j_x.T @ j_x + mu_val * _eye4(t) + 1e-7 * _eye4(t)

        z = torch.linalg.solve(h, xhat)
        j = j_x @ z
        delta = (j * residual).sum() / ((j * j).sum() + 1e-10)

        step = 1.0
        for _ in range(20):
            mu_new = mu - step * delta
            if mu_new.item() > 1e-6:
                break
            step *= 0.5
        mu = (mu - step * delta).detach().clamp(min=1e-6)

        hist["mu"].append(mu.item())
        hist["V"].append((residual**2).sum().item())
        hist["xhat"].append(xhat)
        x_init = xhat

    return hist


def ep_gradient_descent_mu(
    y: torch.Tensor,
    t: torch.Tensor,
    x_star: torch.Tensor,
    mu_init: float = 1.0,
    beta: float = 0.01,
    lr: float = 0.1,
    n_steps: int = 100,
    lower_max_iter: int = 300,
    lower_tol: float = 1e-9,
    progress: bool = False,
) -> dict[str, list]:
    """Update mu with an equilibrium-propagation gradient estimate."""
    mu = torch.tensor(float(mu_init), dtype=t.dtype, device=t.device)
    x_star = x_star.to(dtype=t.dtype, device=t.device)
    x_init = default_x_init(dtype=t.dtype, device=t.device)
    hist = {"mu": [], "V": [], "xhat": []}

    for _ in _iter_range(n_steps, "outer loop step", progress):
        mu_val = mu.item()
        x0, _ = lower_level_gn_ep(
            y,
            mu_val,
            t,
            beta=0.0,
            x_star=x_star,
            max_iter=lower_max_iter,
            tol=lower_tol,
            x_init=x_init,
        )
        x_beta, _ = lower_level_gn_ep(
            y,
            mu_val,
            t,
            beta=beta,
            x_star=x_star,
            max_iter=lower_max_iter,
            tol=lower_tol,
            x_init=x0,
        )

        grad_mu = (0.5 * (x_beta**2).sum() - 0.5 * (x0**2).sum()) / beta

        step = lr
        for _ in range(20):
            mu_new = mu - step * grad_mu
            if mu_new.item() > 1e-6:
                break
            step *= 0.5
        mu = (mu - step * grad_mu).detach().clamp(min=1e-6)

        c, lam = x0[:2], x0[2:]
        residual = y - biexponential(t, c, lam)
        hist["mu"].append(mu.item())
        hist["V"].append((residual**2).sum().item())
        hist["xhat"].append(x0)
        x_init = x0

    return hist


__all__ = [
    "XhatModule",
    "default_x_init",
    "ep_gradient_descent_mu",
    "gauss_newton_mu",
    "gradient_descent_mu",
    "inv_sp",
    "lower_level_gn_ep",
    "sp",
    "sp_d",
]
