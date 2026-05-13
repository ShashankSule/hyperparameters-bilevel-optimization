"""Optimization routines for bilevel biexponential recovery experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares

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


def random_bounded_x_init(
    rng: np.random.Generator,
    t: torch.Tensor,
    *,
    c_min: float = 0.0,
    c_max: float = 1.0,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample a feasible positive `x_init` = [c0, c1, lambda0, lambda1].

    By default amplitudes are sampled in [0,1] and decay/scale parameters are
    sampled from the time-grid range `t` when available.
    """
    if lambda_min is None:
        lambda_min = float(t.min().item()) if isinstance(t, torch.Tensor) else float(min(t))
    if lambda_max is None:
        lambda_max = float(t.max().item()) if isinstance(t, torch.Tensor) else float(max(t))

    c0 = float(rng.uniform(c_min, c_max))
    c1 = float(rng.uniform(c_min, c_max))
    lam0 = float(rng.uniform(lambda_min, lambda_max))
    lam1 = float(rng.uniform(lambda_min, lambda_max))

    arr = np.array([c0, c1, lam0, lam1], dtype=float)
    return torch.tensor(arr, dtype=dtype, device=device)


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


def _to_numpy_1d(value: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(float, copy=False).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def direct_nlls_to_physical(
    z: np.ndarray | list[float],
    *,
    lambda_max: float = 0.05,
) -> np.ndarray:
    """Map box-constrained solver variables [c1, p1, q] to [c1, p1, p2]."""
    z_arr = np.asarray(z, dtype=float)
    c1, p1, q = z_arr
    p2 = p1 + q * (float(lambda_max) - p1)
    return np.array([c1, p1, p2], dtype=float)


def direct_nlls_to_solver(
    params: np.ndarray | list[float],
    *,
    lambda_max: float = 0.05,
) -> np.ndarray:
    """Map physical parameters [c1, p1, p2] to solver variables."""
    c1, p1, p2 = np.asarray(params, dtype=float)
    denom = max(float(lambda_max) - float(p1), 1e-12)
    q = (float(p2) - float(p1)) / denom
    return np.array([c1, p1, np.clip(q, 0.0, 1.0)], dtype=float)


def random_direct_nlls_init(
    rng: np.random.Generator,
    *,
    lambda_min: float = 0.004,
    lambda_max: float = 0.05,
) -> np.ndarray:
    """Draw a feasible random [c1, p1, p2] initialization."""
    c1 = rng.uniform(0.0, 1.0)
    p1 = rng.uniform(float(lambda_min), float(lambda_max))
    q = rng.uniform(0.0, 1.0)
    return direct_nlls_to_physical([c1, p1, q], lambda_max=lambda_max)


def direct_nlls_residual(
    z: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    lambda_max: float = 0.05,
    use_inverses: bool = False,
) -> np.ndarray:
    """Residual for the three-parameter constrained NLLS problem."""
    c1, p1, p2 = direct_nlls_to_physical(z, lambda_max=lambda_max)
    if use_inverses:
        return c1 * np.exp(-t / p1) + (1.0 - c1) * np.exp(-t / p2) - y
    return c1 * np.exp(-p1 * t) + (1.0 - c1) * np.exp(-p2 * t) - y


def direct_nlls_jacobian(
    z: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    lambda_max: float = 0.05,
    use_inverses: bool = False,
) -> np.ndarray:
    """Analytic residual Jacobian with respect to [c1, p1, q]."""
    del y
    c1, p1, q = np.asarray(z, dtype=float)
    p2 = p1 + q * (float(lambda_max) - p1)

    if use_inverses:
        e1 = np.exp(-t / p1)
        e2 = np.exp(-t / p2)
        dg_dp1 = c1 * t * e1 / (p1**2)
        dg_dp2 = (1.0 - c1) * t * e2 / (p2**2)
    else:
        e1 = np.exp(-p1 * t)
        e2 = np.exp(-p2 * t)
        dg_dp1 = -c1 * t * e1
        dg_dp2 = -(1.0 - c1) * t * e2

    dg_dc1 = e1 - e2
    dp2_dp1 = 1.0 - q
    dp2_dq = float(lambda_max) - p1

    return np.column_stack(
        [
            dg_dc1,
            dg_dp1 + dg_dp2 * dp2_dp1,
            dg_dp2 * dp2_dq,
        ]
    )


def solve_direct_nlls(
    y: torch.Tensor | np.ndarray | list[float],
    t: torch.Tensor | np.ndarray | list[float],
    *,
    x_init: np.ndarray | list[float] | None = None,
    seed: int | None = None,
    lambda_min: float = 0.004,
    lambda_max: float = 0.05,
    max_nfev: int | None = 300,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    use_inverses: bool = False,
) -> dict[str, object]:
    """Solve the direct constrained NLLS recovery problem from one initialization."""
    t_np = _to_numpy_1d(t)
    y_np = _to_numpy_1d(y)
    if t_np.shape != y_np.shape:
        raise ValueError(f"t and y must have the same shape, got {t_np.shape} and {y_np.shape}.")

    rng = np.random.default_rng(seed)
    if x_init is None:
        x_init_arr = random_direct_nlls_init(rng, lambda_min=lambda_min, lambda_max=lambda_max)
    else:
        x_init_arr = np.asarray(x_init, dtype=float).reshape(3)

    lower = np.array([0.0, float(lambda_min), 0.0], dtype=float)
    upper = np.array([1.0, float(lambda_max), 1.0], dtype=float)
    z0 = np.clip(direct_nlls_to_solver(x_init_arr, lambda_max=lambda_max), lower, upper)
    result = least_squares(
        lambda z: direct_nlls_residual(z, t_np, y_np, lambda_max=lambda_max, use_inverses=use_inverses),
        z0,
        jac=lambda z: direct_nlls_jacobian(z, t_np, y_np, lambda_max=lambda_max, use_inverses=use_inverses),
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )
    x_hat = direct_nlls_to_physical(result.x, lambda_max=lambda_max)
    residual = np.asarray(result.fun, dtype=float)
    rss = float(np.sum(residual**2))
    return {
        "x_init": x_init_arr,
        "x_hat": x_hat,
        "z_hat": result.x,
        "residual": residual,
        "rss": rss,
        "cost": float(result.cost),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if result.njev is not None else None,
        "optimality": float(result.optimality),
        "active_mask": result.active_mask.astype(int).tolist(),
    }


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
    init_seed: int | None = None,
) -> dict[str, list]:
    """Optimize log_mu with a manually computed implicit-differentiation gradient."""
    log_mu = torch.tensor(float(np.log(mu_init)), dtype=t.dtype, device=t.device)
    if init_seed is not None:
        rng = np.random.default_rng(int(init_seed))
        x_init = random_bounded_x_init(rng, t, dtype=t.dtype, device=t.device)
    else:
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
    init_seed: int | None = None,
) -> dict[str, list]:
    """Optimize mu with a scalar Gauss-Newton step and positivity line search."""
    mu = torch.tensor(float(mu_init), dtype=t.dtype, device=t.device)
    if init_seed is not None:
        rng = np.random.default_rng(int(init_seed))
        x_init = random_bounded_x_init(rng, t, dtype=t.dtype, device=t.device)
    else:
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
    init_seed: int | None = None,
) -> dict[str, list]:
    """Update mu with an equilibrium-propagation gradient estimate."""
    mu = torch.tensor(float(mu_init), dtype=t.dtype, device=t.device)
    x_star = x_star.to(dtype=t.dtype, device=t.device)
    if init_seed is not None:
        rng = np.random.default_rng(int(init_seed))
        x_init = random_bounded_x_init(rng, t, dtype=t.dtype, device=t.device)
    else:
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
    "direct_nlls_jacobian",
    "direct_nlls_residual",
    "direct_nlls_to_physical",
    "direct_nlls_to_solver",
    "ep_gradient_descent_mu",
    "gauss_newton_mu",
    "gradient_descent_mu",
    "inv_sp",
    "lower_level_gn_ep",
    "random_direct_nlls_init",
    "solve_direct_nlls",
    "sp",
    "sp_d",
]
